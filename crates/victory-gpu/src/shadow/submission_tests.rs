use super::*;

#[test]
fn repeated_gpu_writes_keep_their_first_copy_back_order() {
    let Some(host) = host() else { return };
    let mut title = [0u8; 3];
    let start = title.as_mut_ptr() as usize;
    let make_shadow = || {
        let len = title.len() as u64;
        let buffer = host.default_buffer(len).unwrap();
        Arc::new(Shadow::new(buffer, 0, start as u64, len))
    };
    let a = make_shadow();
    let b = make_shadow();
    let first = host.new_list().unwrap();
    first.defer_refresh(&a, start, 3);
    first.defer_refresh(&b, start, 3);
    first.defer_gpu_write(&a, start, 2);
    first.defer_gpu_write(&a, start, 2);
    first.defer_gpu_write(&a, start, 1);
    first.close();

    let (up_a, ptr_a) = host.upload_buffer(1).unwrap();
    let (up_b, ptr_b) = host.upload_buffer(1).unwrap();

    let second = host.new_list().unwrap();
    second.defer_gpu_write(&b, start, 2);
    second.defer(move || {
        // SAFETY: the title array lives through both synchronous submissions.
        unsafe { (start as *mut u8).add(2).write(33) };
    });
    second.defer_refresh(&a, start + 2, 1);
    second.defer_gpu_write(&a, start, 2);
    let a_check = a.clone();
    let b_check = b.clone();
    second.defer_host(move || {
        assert_eq!(
            *a_check.pending.lock().unwrap(),
            [(start, start + 2), (start, start + 1)]
        );
        assert_eq!(*b_check.pending.lock().unwrap(), [(start, start + 2)]);
    });
    // Record GPU writes to byte 0 of `a` (+11) and `b` (+22) via CopyBufferRegion.
    // SAFETY: open list and live buffers.
    unsafe {
        second.list.CopyBufferRegion(&a.buffer, 0, &up_a, 0, 1);
        second.list.CopyBufferRegion(&b.buffer, 0, &up_b, 0, 1);
    }
    second.close();

    for expected in [22u8, 44u8] {
        // SAFETY: mapped 1-byte upload buffers.
        unsafe {
            ptr_a.write(expected - 11);
            ptr_b.write(expected);
        }
        host.execute(&[&first, &second]);
        assert_eq!(
            title,
            [expected, 0, 33],
            "the second shadow writes back last"
        );
    }
}
