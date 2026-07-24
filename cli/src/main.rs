use clap::Parser;
use owo_colors::OwoColorize;
use std::ffi::CString;
use std::io::{BufRead, BufReader, Read, Write};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

use pngstreamenc::{
    close_png_encode, encode_scanlines, free_error_string_enc, get_last_error_enc, open_png_encode,
    open_png_encode_stream, ZlibOptions,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Stream PPM (P6) / PAM (P7) to PNG with advanced ZLIB & Filtering controls")]
struct Args {
    #[arg(short, long, default_value_t = 6, help = "ZLIB compression level (0-9)")]
    level: i32,

    #[arg(
    long,
    default_value = "default",
    help = "ZLIB strategy: default, filtered, huffman, rle, fixed"
    )]
    strategy: String,

    #[arg(
    short = 'f',
    long,
    default_value = "adaptive",
    help = "PNG filter type: none (0), sub (1), up (2), avg (3), paeth (4), adaptive (5)"
    )]
    filter: String,

    #[arg(long, default_value_t = 15, help = "ZLIB window bits (8-15)")]
    window_bits: i32,

    #[arg(long, default_value_t = 8, help = "ZLIB memory level (1-9)")]
    mem_level: i32,

    #[arg(short = 's', long, default_value_t = 250, help = "Number of scanlines to process per chunk")]
    scanlines: u32,

    #[arg(help = "Input PPM/PAM file (use '-' for stdin)")]
    input: String,

    #[arg(help = "Output PNG file (use '-' for stdout)")]
    output: String,
}

unsafe extern "C" fn stdout_write_cb(_user_data: *mut std::ffi::c_void, buf: *const u8, len: usize) -> usize {
    let slice = unsafe { std::slice::from_raw_parts(buf, len) };
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    if handle.write_all(slice).is_ok() { len } else { 0 }
}

fn read_token<R: BufRead>(reader: &mut R) -> std::io::Result<String> {
    let mut token = String::new();
    let mut in_comment = false;

    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            if !token.is_empty() {
                return Ok(token);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Unexpected EOF reading header",
            ));
        }

        let mut consumed = 0;
        for &byte in buf {
            consumed += 1;
            let c = byte as char;

            if in_comment {
                if c == '\n' {
                    in_comment = false;
                }
                continue;
            }

            if c == '#' && token.is_empty() {
                in_comment = true;
                continue;
            }

            if c.is_ascii_whitespace() {
                if !token.is_empty() {
                    reader.consume(consumed);
                    return Ok(token);
                }
            } else {
                token.push(c);
            }
        }

        reader.consume(consumed);
    }
}

fn read_image_header<R: BufRead>(reader: &mut R) -> std::io::Result<(u32, u32, u8, u8)> {
    let magic = read_token(reader)?;

    if magic == "P6" {
        let w_str = read_token(reader)?;
        let h_str = read_token(reader)?;
        let maxval_str = read_token(reader)?;

        let w: u32 = w_str.parse().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid PPM width")
        })?;
        let h: u32 = h_str.parse().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid PPM height")
        })?;
        let maxval: u32 = maxval_str.parse().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid PPM maxval")
        })?;

        let bit_depth = if maxval > 255 { 16 } else { 8 };
        let color_type = 2; // RGB
        Ok((w, h, bit_depth, color_type))
    } else if magic == "P7" {
        let mut w: Option<u32> = None;
        let mut h: Option<u32> = None;
        let mut depth: Option<u32> = None;
        let mut maxval: Option<u32> = None;

        loop {
            let token = read_token(reader)?;
            if token == "ENDHDR" {
                break;
            }
            match token.as_str() {
                "WIDTH" => {
                    let val = read_token(reader)?;
                    w = Some(val.parse().map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid PAM WIDTH")
                    })?);
                }
                "HEIGHT" => {
                    let val = read_token(reader)?;
                    h = Some(val.parse().map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid PAM HEIGHT")
                    })?);
                }
                "DEPTH" => {
                    let val = read_token(reader)?;
                    depth = Some(val.parse().map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid PAM DEPTH")
                    })?);
                }
                "MAXVAL" => {
                    let val = read_token(reader)?;
                    maxval = Some(val.parse().map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid PAM MAXVAL")
                    })?);
                }
                "TUPLTYPE" => {
                    let _val = read_token(reader)?;
                }
                _ => {}
            }
        }

        let width = w.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "PAM missing WIDTH")
        })?;
        let height = h.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "PAM missing HEIGHT")
        })?;
        let depth_val = depth.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "PAM missing DEPTH")
        })?;
        let maxval_val = maxval.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "PAM missing MAXVAL")
        })?;

        let bit_depth = if maxval_val > 255 { 16 } else { 8 };

        let color_type = match depth_val {
            1 => 0, // Grayscale
            2 => 4, // Grayscale + Alpha
            3 => 2, // RGB
            4 => 6, // RGBA
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unsupported PAM DEPTH: {}", depth_val),
                ));
            }
        };

        Ok((width, height, bit_depth, color_type))
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Expected P6 (PPM) or P7 (PAM) format, got: {}", magic),
        ))
    }
}

fn print_last_error() {
    let err_ptr = get_last_error_enc();
    if !err_ptr.is_null() {
        let c_str = unsafe { std::ffi::CStr::from_ptr(err_ptr) };
        eprintln!("{}", format!("Error: {}", c_str.to_string_lossy()).bright_red());
        free_error_string_enc(err_ptr as *mut _);
    } else {
        eprintln!("{}", "Error: Encoding failed (no further details provided).".bright_red());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let strategy_val = match args.strategy.to_lowercase().as_str() {
        "default" => 0,
        "filtered" => 1,
        "huffman" => 2,
        "rle" => 3,
        "fixed" => 4,
        _ => {
            eprintln!("{}", "Error: Invalid ZLIB strategy".bright_red());
            std::process::exit(1);
        }
    };

    let filter_val = match args.filter.to_lowercase().as_str() {
        "none" | "0" => 0u8,
        "sub" | "1" => 1u8,
        "up" | "2" => 2u8,
        "avg" | "average" | "3" => 3u8,
        "paeth" | "4" => 4u8,
        "adaptive" | "5" => 5u8,
        _ => {
            eprintln!("{}", "Error: Invalid PNG filter type".bright_red());
            std::process::exit(1);
        }
    };

    let options = ZlibOptions {
        level: args.level,
        strategy: strategy_val,
        window_bits: args.window_bits,
        mem_level: args.mem_level,
        max_idat_size: 8388608,
        expected_idat_size: 0,
    };

    let raw_input: Box<dyn Read> = if args.input == "-" {
        Box::new(std::io::stdin())
    } else {
        let file = std::fs::File::open(&args.input)?;

        // Add this hint to trigger kernel-level sequential read-ahead
        #[cfg(unix)]
        {
            let fd = file.as_raw_fd();
            unsafe {
                // Advise the kernel we want to read the file from start to finish
                libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL);
            }
        }

        Box::new(file)
    };

    // Scaled buffer capacity to 4MB to prevent excessive sys-calls during read
    let mut input = BufReader::with_capacity(4 * 1024 * 1024, raw_input);

    let (width, height, bit_depth, color_type) = match read_image_header(&mut input) {
        Ok(meta) => meta,
        Err(e) => {
            eprintln!("{}", format!("Failed to read input header: {}", e).bright_red());
            std::process::exit(1);
        }
    };

    eprintln!(
        "Encoding: {}x{} | Strategy: {} | Level: {} | BitDepth: {} | ColorType: {} | Filter: {}",
        width, height, strategy_val, args.level, bit_depth, color_type, filter_val
    );

    let handle = if args.output == "-" {
        open_png_encode_stream(
            stdout_write_cb,
            std::ptr::null_mut(),
                               width,
                               height,
                               bit_depth,
                               color_type,
                               filter_val,
                               options,
        )
    } else {
        let c_output = CString::new(args.output).unwrap();
        open_png_encode(c_output.as_ptr(), width, height, bit_depth, color_type, filter_val, options)
    };

    if handle.is_null() {
        unsafe {
            let err_ptr = get_last_error_enc();
            if !err_ptr.is_null() {
                let c_str = std::ffi::CStr::from_ptr(err_ptr);
                eprintln!("{}", format!("Error initializing encoder: {}", c_str.to_string_lossy()).bright_red());
                free_error_string_enc(err_ptr as *mut _);
            }
        }
        std::process::exit(1);
    }

    let channels = match color_type {
        0 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        _ => 3,
    };

    let bytes_per_scanline = (width as usize * channels * bit_depth as usize) / 8;
    let chunk_bytes = bytes_per_scanline * args.scanlines as usize;
    let mut buffer = vec![0u8; chunk_bytes];
    let mut current_y = 0;
    let mut success = true;

    let stderr = std::io::stderr();
    let mut stderr_handle = stderr.lock();

    while current_y < height {
        let lines_to_read = std::cmp::min(args.scanlines, height - current_y);
        let bytes_to_read = lines_to_read as usize * bytes_per_scanline;

        if let Err(e) = input.read_exact(&mut buffer[..bytes_to_read]) {
            let _ = writeln!(stderr_handle, "\nError reading input stream: {}", e);
            success = false;
            break;
        }

        if !encode_scanlines(handle, buffer.as_ptr(), lines_to_read) {
            let _ = writeln!(stderr_handle, "\nError encountered during scanline encoding.");
            success = false;
            break;
        }

        current_y += lines_to_read;

        let _ = write!(stderr_handle, "\rEncoded {}/{} scanlines...", current_y, height);
    }

    let closed = close_png_encode(handle);

    if success && closed {
        let _ = writeln!(stderr_handle, "\nSuccess!");
        Ok(())
    } else {
        let _ = writeln!(stderr_handle);
        print_last_error();
        Err("Command failed".into())
    }
}
