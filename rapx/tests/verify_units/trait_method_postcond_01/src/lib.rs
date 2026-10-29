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
