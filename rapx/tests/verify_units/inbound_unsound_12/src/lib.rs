#![feature(register_tool)]
#![register_tool(rapx)]
#![allow(dead_code)]

// UNSOUND: the range start is guarded, but the end can be outside the slice.
#[rapx::verify]
pub fn unsound_std_range_missing_end_guard(data: &[u32], start: usize, end: usize) -> usize {
    if start <= data.len() {
        unsafe { data.get_unchecked(start..end).len() }
    } else {
        0
    }
}

#[rapx::verify]
pub fn unsound_std_range_inverted(data: &[u32], start: usize, end: usize) -> usize {
    if end <= data.len() {
        unsafe { data.get_unchecked(start..end).len() }
    } else {
        0
    }
}
