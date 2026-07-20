use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::MaybeUninit;
use std::os::raw::c_char;
use std::ptr;

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.to_string());
}

const PNG_MAGIC: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

#[repr(C)]
pub struct ZlibOptions {
    pub level: i32,
    pub strategy: i32,
    pub window_bits: i32,
    pub mem_level: i32,
}

pub type PngWriteCallback = unsafe extern "C" fn(user_data: *mut std::ffi::c_void, buf: *const u8, len: usize) -> usize;

struct FfiWriter {
    cb: PngWriteCallback,
    user_data: *mut std::ffi::c_void,
}

impl Write for FfiWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let bytes_written = unsafe { (self.cb)(self.user_data, buf.as_ptr(), buf.len()) };
        if bytes_written == 0 && !buf.is_empty() {
            Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "FFI callback wrote 0 bytes"))
        } else {
            Ok(bytes_written)
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct PngEncoder {
    writer: BufWriter<Box<dyn Write>>,
    zstream: Box<libz_sys::z_stream>,
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: u8,
    pub filter_type: u8, // 0: None, 1: Sub, 2: Up, 3: Avg, 4: Paeth, 5: Adaptive
    pub bytes_per_scanline: usize,
    pub bpp: usize,
    out_buf: Vec<u8>,
    row_buf: Vec<u8>,
    prev_row: Vec<u8>,
    temp_row: Vec<u8>,
}

fn zlib_crc32(crc: u32, buf: &[u8]) -> u32 {
    unsafe {
        libz_sys::crc32(crc as std::os::raw::c_ulong, buf.as_ptr(), buf.len() as std::os::raw::c_uint) as u32
    }
}

fn write_chunk(writer: &mut dyn Write, chunk_type: &[u8; 4], data: &[u8]) -> std::io::Result<()> {
    let mut header = [0u8; 8];
    header[0..4].copy_from_slice(&(data.len() as u32).to_be_bytes());
    header[4..8].copy_from_slice(chunk_type);

    let crc = zlib_crc32(0, chunk_type);
    let crc = if !data.is_empty() { zlib_crc32(crc, data) } else { crc };

    writer.write_all(&header)?;
    if !data.is_empty() {
        writer.write_all(data)?;
    }
    writer.write_all(&crc.to_be_bytes())?;
    Ok(())
}

fn write_idat(enc: &mut PngEncoder) -> Result<(), String> {
    let produced = enc.out_buf.len() - enc.zstream.avail_out as usize;
    if produced > 0 {
        write_chunk(&mut enc.writer, b"IDAT", &enc.out_buf[..produced])
        .map_err(|_| "Failed to write IDAT chunk".to_string())?;
    }
    enc.zstream.next_out = enc.out_buf.as_mut_ptr();
    enc.zstream.avail_out = enc.out_buf.len() as u32;
    Ok(())
}

// High performance filter implementations
#[inline]
fn apply_filter_none(curr: &[u8], out: &mut [u8]) -> u64 {
    out[0] = 0;
    let len = curr.len();
    out[1..1 + len].copy_from_slice(curr);
    curr.iter().fold(0u64, |acc, &b| acc + (b as i8).unsigned_abs() as u64)
}

#[inline]
fn apply_filter_sub(curr: &[u8], bpp: usize, out: &mut [u8]) -> u64 {
    out[0] = 1;
    let len = curr.len();
    let out_payload = &mut out[1..1 + len];
    let mut sad = 0u64;

    let first = bpp.min(len);
    for i in 0..first {
        let val = curr[i];
        out_payload[i] = val;
        sad += val as u64;
    }

    for i in first..len {
        let val = curr[i].wrapping_sub(curr[i - bpp]);
        out_payload[i] = val;
        sad += val as u64;
    }
    sad
}

#[inline]
fn apply_filter_up(curr: &[u8], prev: &[u8], out: &mut [u8]) -> u64 {
    out[0] = 2;
    let len = curr.len();
    let out_payload = &mut out[1..1 + len];
    let mut sad = 0u64;

    for i in 0..len {
        let val = curr[i].wrapping_sub(prev[i]);
        out_payload[i] = val;
        sad += val as u64;
    }
    sad
}

#[inline]
fn apply_filter_avg(curr: &[u8], prev: &[u8], bpp: usize, out: &mut [u8]) -> u64 {
    out[0] = 3;
    let len = curr.len();
    let out_payload = &mut out[1..1 + len];
    let mut sad = 0u64;

    let first = bpp.min(len);
    for i in 0..first {
        let avg = (prev[i] / 2) as u8;
        let val = curr[i].wrapping_sub(avg);
        out_payload[i] = val;
        sad += val as u64;
    }

    for i in first..len {
        let a = curr[i - bpp] as u16;
        let b = prev[i] as u16;
        let avg = ((a + b) / 2) as u8;
        let val = curr[i].wrapping_sub(avg);
        out_payload[i] = val;
        sad += val as u64;
    }
    sad
}

#[inline]
fn paeth_predict(a: u8, b: u8, c: u8) -> u8 {
    let a_i = a as i16;
    let b_i = b as i16;
    let c_i = c as i16;
    let p = a_i + b_i - c_i;
    let pa = (p - a_i).abs();
    let pb = (p - b_i).abs();
    let pc = (p - c_i).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

#[inline]
fn apply_filter_paeth(curr: &[u8], prev: &[u8], bpp: usize, out: &mut [u8]) -> u64 {
    out[0] = 4;
    let len = curr.len();
    let out_payload = &mut out[1..1 + len];
    let mut sad = 0u64;

    let first = bpp.min(len);
    for i in 0..first {
        let val = curr[i].wrapping_sub(paeth_predict(0, prev[i], 0));
        out_payload[i] = val;
        sad += (val as i8).unsigned_abs() as u64;
    }

    for i in first..len {
        let a = curr[i - bpp];
        let b = prev[i];
        let c = prev[i - bpp];
        let val = curr[i].wrapping_sub(paeth_predict(a, b, c));
        out_payload[i] = val;
        sad += (val as i8).unsigned_abs() as u64;
    }
    sad
}

fn apply_filter_adaptive(curr: &[u8], prev: &[u8], bpp: usize, out: &mut [u8], temp: &mut [u8]) {
    let payload_len = 1 + curr.len();

    let mut best_sad = apply_filter_none(curr, temp);
    out[..payload_len].copy_from_slice(&temp[..payload_len]);

    let sad1 = apply_filter_sub(curr, bpp, temp);
    if sad1 < best_sad {
        best_sad = sad1;
        out[..payload_len].copy_from_slice(&temp[..payload_len]);
    }

    let sad2 = apply_filter_up(curr, prev, temp);
    if sad2 < best_sad {
        best_sad = sad2;
        out[..payload_len].copy_from_slice(&temp[..payload_len]);
    }

    let sad3 = apply_filter_avg(curr, prev, bpp, temp);
    if sad3 < best_sad {
        best_sad = sad3;
        out[..payload_len].copy_from_slice(&temp[..payload_len]);
    }

    let sad4 = apply_filter_paeth(curr, prev, bpp, temp);
    if sad4 < best_sad {
        out[..payload_len].copy_from_slice(&temp[..payload_len]);
    }
}

impl PngEncoder {
    fn new(
        mut writer: Box<dyn Write>,
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: u8,
        filter_type: u8,
        options: ZlibOptions,
    ) -> Result<Self, String> {
        writer.write_all(&PNG_MAGIC).map_err(|_| "Failed to write PNG magic".to_string())?;

        let mut ihdr = [0u8; 13];
        ihdr[0..4].copy_from_slice(&width.to_be_bytes());
        ihdr[4..8].copy_from_slice(&height.to_be_bytes());
        ihdr[8] = bit_depth;
        ihdr[9] = color_type;
        ihdr[10] = 0;
        ihdr[11] = 0;
        ihdr[12] = 0;

        write_chunk(&mut writer, b"IHDR", &ihdr).map_err(|_| "Failed to write IHDR".to_string())?;

        let channels = match color_type {
            0 => 1,
            2 => 3,
            3 => 1,
            4 => 2,
            6 => 4,
            _ => return Err("Unsupported color type".to_string()),
        };

        let bytes_per_scanline = (width as usize * (channels * bit_depth as usize) + 7) / 8;
        let bpp = std::cmp::max(1, (channels * bit_depth as usize) / 8);

        let mut zstream_uninit = Box::new(MaybeUninit::<libz_sys::z_stream>::zeroed());
        let z_err = unsafe {
            libz_sys::deflateInit2_(
                zstream_uninit.as_mut_ptr(),
                                    options.level,
                                    libz_sys::Z_DEFLATED,
                                    options.window_bits,
                                    options.mem_level,
                                    options.strategy,
                                    libz_sys::zlibVersion(),
                                    std::mem::size_of::<libz_sys::z_stream>() as i32,
            )
        };

        if z_err != libz_sys::Z_OK {
            return Err(format!("ZLIB Init failed with code: {}", z_err));
        }

        let mut zstream: Box<libz_sys::z_stream> = unsafe {
            let raw = Box::into_raw(zstream_uninit);
            Box::from_raw(raw as *mut libz_sys::z_stream)
        };

        // Increase buffer sizes to avoid voluntary context switching from constant flushing
        let mut out_buf = vec![0u8; 2 * 1024 * 1024];
        zstream.next_out = out_buf.as_mut_ptr();
        zstream.avail_out = out_buf.len() as u32;

        Ok(Self {
            writer: BufWriter::with_capacity(1024 * 1024, writer),
           zstream,
           width,
           height,
           bit_depth,
           color_type,
           filter_type,
           bytes_per_scanline,
           bpp,
           out_buf,
           row_buf: vec![0u8; bytes_per_scanline + 1],
           prev_row: vec![0u8; bytes_per_scanline],
           temp_row: vec![0u8; bytes_per_scanline + 1],
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn open_png_encode(
    filename: *const c_char,
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    filter_type: u8,
    options: ZlibOptions,
) -> *mut PngEncoder {
    let c_str = unsafe { CStr::from_ptr(filename) };
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let file = match File::create(path) {
        Ok(f) => f,
        Err(_) => {
            set_last_error("Could not create output file");
            return ptr::null_mut();
        }
    };

    match PngEncoder::new(Box::new(file), width, height, bit_depth, color_type, filter_type, options) {
        Ok(encoder) => Box::into_raw(Box::new(encoder)),
        Err(e) => {
            set_last_error(&e);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn open_png_encode_stream(
    write_cb: PngWriteCallback,
    user_data: *mut std::ffi::c_void,
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    filter_type: u8,
    options: ZlibOptions,
) -> *mut PngEncoder {
    let writer = Box::new(FfiWriter { cb: write_cb, user_data });

    match PngEncoder::new(writer, width, height, bit_depth, color_type, filter_type, options) {
        Ok(encoder) => Box::into_raw(Box::new(encoder)),
        Err(e) => {
            set_last_error(&e);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn encode_scanlines(
    handle: *mut PngEncoder,
    data: *const u8,
    num_scanlines: u32,
) -> bool {
    if handle.is_null() || data.is_null() || num_scanlines == 0 {
        set_last_error("Invalid encoder handle, null data buffer, or 0 scanlines specified");
        return false;
    }

    let enc = unsafe { &mut *handle };
    let bytes_per_line = enc.bytes_per_scanline;
    let data_slice = unsafe { std::slice::from_raw_parts(data, (num_scanlines as usize) * bytes_per_line) };

    for i in 0..num_scanlines as usize {
        let src_start = i * bytes_per_line;
        let row_data = &data_slice[src_start..src_start + bytes_per_line];

        match enc.filter_type {
            0 => { apply_filter_none(row_data, &mut enc.row_buf); }
            1 => { apply_filter_sub(row_data, enc.bpp, &mut enc.row_buf); }
            2 => { apply_filter_up(row_data, &enc.prev_row, &mut enc.row_buf); }
            3 => { apply_filter_avg(row_data, &enc.prev_row, enc.bpp, &mut enc.row_buf); }
            4 => { apply_filter_paeth(row_data, &enc.prev_row, enc.bpp, &mut enc.row_buf); }
            5 => apply_filter_adaptive(row_data, &enc.prev_row, enc.bpp, &mut enc.row_buf, &mut enc.temp_row),
            _ => { apply_filter_none(row_data, &mut enc.row_buf); }
        }

        enc.prev_row.copy_from_slice(row_data);

        enc.zstream.next_in = enc.row_buf.as_mut_ptr();
        enc.zstream.avail_in = enc.row_buf.len() as u32;

        while enc.zstream.avail_in > 0 {
            if enc.zstream.avail_out == 0 {
                if let Err(e) = write_idat(enc) {
                    set_last_error(&format!("IO Write Error during IDAT flush: {}", e));
                    return false;
                }
            }
            let z_res = unsafe { libz_sys::deflate(enc.zstream.as_mut(), libz_sys::Z_NO_FLUSH) };
            if z_res != libz_sys::Z_OK {
                set_last_error(&format!("ZLIB deflate error code: {}", z_res));
                return false;
            }
        }
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn close_png_encode(handle: *mut PngEncoder) -> bool {
    if handle.is_null() {
        return false;
    }
    let mut enc = unsafe { Box::from_raw(handle) };

    enc.zstream.next_in = ptr::null_mut();
    enc.zstream.avail_in = 0;

    loop {
        if enc.zstream.avail_out == 0 {
            if let Err(e) = write_idat(&mut enc) {
                set_last_error(&format!("IO Write Error during final stream flush: {}", e));
                return false;
            }
        }
        let ret = unsafe { libz_sys::deflate(enc.zstream.as_mut(), libz_sys::Z_FINISH) };
        if ret == libz_sys::Z_STREAM_END {
            break;
        } else if ret != libz_sys::Z_OK && ret != libz_sys::Z_BUF_ERROR {
            set_last_error(&format!("ZLIB deflate finish error code: {}", ret));
            return false;
        }
    }

    if let Err(e) = write_idat(&mut enc) {
        set_last_error(&format!("IO Write Error writing remaining IDAT data: {}", e));
        return false;
    }

    if let Err(e) = write_chunk(&mut enc.writer, b"IEND", &[]) {
        set_last_error(&format!("IO Write Error writing IEND chunk: {}", e));
        return false;
    }

    if let Err(e) = enc.writer.flush() {
        set_last_error(&format!("IO Flush Error: {}", e));
        return false;
    }

    unsafe { libz_sys::deflateEnd(enc.zstream.as_mut()) };
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        let err = e.borrow();
        let sanitized: Vec<u8> = err.bytes().filter(|&b| b != 0).collect();
        match CString::new(sanitized) {
            Ok(c_str) => c_str.into_raw(),
                    Err(_) => ptr::null(),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn free_error_string_enc(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { let _ = CString::from_raw(ptr); }
    }
}
