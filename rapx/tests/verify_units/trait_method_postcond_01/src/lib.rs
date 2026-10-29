#![feature(register_tool)]
#![register_tool(rapx)]
#![allow(dead_code)]

unsafe trait SliceIndexLike {
    #[rapx::requires(NonNull(slice))]
    #[rapx::ensures(NonNull(return))]
    unsafe fn get_unchecked(self, slice: *const [u8]) -> *const u8;
}

pub struct GoodIndex;
pub struct BadIndex;

#[rapx::verify]
unsafe impl SliceIndexLike for GoodIndex {
    unsafe fn get_unchecked(self, slice: *const [u8]) -> *const u8 {
        slice as *const u8
    }
}

#[rapx::verify]
unsafe impl SliceIndexLike for BadIndex {
    unsafe fn get_unchecked(self, _: *const [u8]) -> *const u8 {
        core::ptr::null()
    }
}

#[rapx::verify]
#[rapx::requires(NonNull(slice))]
pub unsafe fn call_bad_index(slice: *const [u8]) {
    let ptr = unsafe { BadIndex.get_unchecked(slice) };
    let _ = unsafe { core::ptr::NonNull::new_unchecked(ptr as *mut u8) };
}

#[rapx::verify]
#[rapx::requires(NonNull(slice))]
pub unsafe fn call_generic_index<I: SliceIndexLike>(index: I, slice: *const [u8]) {
    let ptr = unsafe { index.get_unchecked(slice) };
    let _ = unsafe { core::ptr::NonNull::new_unchecked(ptr as *mut u8) };
}

unsafe trait OffsetIndex {
    #[rapx::ensures(SliceOffset(return, slice, index))]
    unsafe fn offset(index: usize, slice: *const [u8]) -> *const u8;
}

pub struct GoodOffset;
pub struct BadOffset;

#[rapx::verify]
unsafe impl OffsetIndex for GoodOffset {
    unsafe fn offset(index: usize, slice: *const [u8]) -> *const u8 {
        unsafe { slice.cast::<u8>().add(index) }
    }
}

#[rapx::verify]
unsafe impl OffsetIndex for BadOffset {
    unsafe fn offset(_: usize, slice: *const [u8]) -> *const u8 {
        slice.cast::<u8>()
    }
}

unsafe trait RangeOffsetIndex {
    #[rapx::ensures(SliceOffset(return, slice, index))]
    unsafe fn offset(index: core::ops::Range<usize>, slice: *const [u8]) -> *const [u8];
}

pub struct BadRangeLength;

#[rapx::verify]
unsafe impl RangeOffsetIndex for BadRangeLength {
    unsafe fn offset(index: core::ops::Range<usize>, slice: *const [u8]) -> *const [u8] {
        let start = unsafe { slice.cast::<u8>().add(index.start) };
        core::ptr::slice_from_raw_parts(start, index.end - index.start + 1)
    }
}
