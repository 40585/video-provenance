use serde::{Deserialize, Serialize};
use std::cmp::{max, min};

// Adopted from:
// https://github.com/FFmpeg/FFmpeg/blob/d4966f0a7484afb71239690d3591cb4d18af4557/libavfilter/vf_avgblur.c

#[derive(Serialize, Deserialize, Debug)]
pub struct AvgBlurContext {
    // Inputs
    pub radius: i32,
    pub radius_v: i32,

    // Implementation-level things, currently unused.
    pub planes: i32, // Always 7 in our case
    pub depth: i32,  // Always 8 in our case
    pub max: i32,    // Always 255 in our case

    // We limit to the case where width/height is the same for each color channel unlike ffmpeg.
    // All 3 are derived from the input image.
    pub area: i32,
    pub width: i32,
    pub height: i32,
}

impl AvgBlurContext {
    pub fn new(radius: i32, radius_v: i32, image_width: i32, image_height: i32) -> Option<Self> {
        let mut context = AvgBlurContext {
            radius,
            radius_v,
            planes: 7, // Bitmask for planes to filter
            depth: 8,  // Implementing the 8-bit color channels version
            max: 255,
            area: (2 * radius + 1) * (2 * radius_v + 1),
            width: image_width,
            height: image_height,
        };
        Some(context)
    }

    pub fn avgblur(c: &AvgBlurContext, src: &[u8], dst: &mut [u8]) {
        // These are basically free, doing this to make analogy with ffmpeg easier
        let (height, width) = (c.height, c.width);
        let mut col_sum = vec![0i32; (width + (1024 * 2 + 1)) as usize];
        let (linesize, dlinesize) = (c.width, c.width);
        let mut sum = 0;
        let (size_w, size_h) = (c.radius, c.radius_v);
        let area = c.area;

        for x in -size_w..0 {
            sum = (src[0] as i32) * size_h;
            for y in 0..size_h + 1 {
                sum += src[(y * linesize) as usize] as i32;
            }
            col_sum[(size_w + x) as usize] = sum;
        }

        for x in 0..width {
            sum = (src[x as usize] as i32) * size_h;
            for y in 0..size_h + 1 {
                sum += src[(x + y * linesize) as usize] as i32;
            }
            col_sum[(size_w + x) as usize] = sum;
        }

        for x in width..(width + size_w) {
            sum = (src[(width - 1) as usize] as i32) * size_h;
            for y in 0..size_h + 1 {
                sum += src[(width - 1 + y * linesize) as usize] as i32;
            }
            col_sum[(size_w + x) as usize] = sum;
        }

        sum = 0;
        for x in -size_w..size_w + 1 {
            sum += col_sum[(size_w + x) as usize];
        }
        dst[0] = (sum / area) as u8;

        for x in 1..width {
            sum = (sum - col_sum[(x - 1) as usize] + col_sum[(x + 2 * size_w) as usize]);
            dst[x as usize] = (sum / area) as u8;
        }

        for y in 1..height {
            let syp = min(size_h, height - y - 1) * linesize;
            let syn = min(y, size_h + 1) * linesize;

            sum = 0;

            for x in -size_w..0 {
                col_sum[(size_w + x) as usize] += (src[(y * linesize + syp) as usize]
                    - src[(y * linesize - syn) as usize])
                    as i32;
            }

            for x in 0..width {
                col_sum[(size_w + x) as usize] += (src[(y * linesize + x + syp) as usize]
                    - src[(y * linesize + x - syn) as usize])
                    as i32;
            }

            for x in width..(width + size_w) {
                col_sum[(size_w + x) as usize] += (src[(y * linesize + width - 1 + syp) as usize]
                    - src[(y * linesize + width - 1 - syn) as usize])
                    as i32;
            }

            for x in -size_w..(size_w + 1) {
                sum += col_sum[(size_w + x) as usize];
            }
            dst[(y * dlinesize) as usize] = (sum / area) as u8;

            for x in 1..width {
                sum = sum - col_sum[(x - 1) as usize] + col_sum[(x + 2 * size_w) as usize];
                dst[(x + y * dlinesize) as usize] = (sum / area) as u8;
            }
        }
    }
}
