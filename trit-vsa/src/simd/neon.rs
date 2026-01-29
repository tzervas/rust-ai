//! ARM NEON SIMD implementations for aarch64.

#![cfg(all(feature = "simd", target_arch = "aarch64"))]

use crate::packed::PackedTritVec;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-accelerated dot product.
///
/// # Safety
///
/// NEON is always available on aarch64, but this function uses unsafe intrinsics.
pub unsafe fn dot_neon(a: &PackedTritVec, b: &PackedTritVec) -> i32 {
    let a_plus = a.plus_plane();
    let a_minus = a.minus_plane();
    let b_plus = b.plus_plane();
    let b_minus = b.minus_plane();

    let mut pp_sum: i32 = 0;
    let mut mm_sum: i32 = 0;
    let mut pm_sum: i32 = 0;
    let mut mp_sum: i32 = 0;

    // Process 4 u32s (128 bits) at a time
    let chunks = a_plus.len() / 4;

    for i in 0..chunks {
        let idx = i * 4;

        // Load 128 bits from each plane
        let ap = vld1q_u32(a_plus[idx..].as_ptr());
        let am = vld1q_u32(a_minus[idx..].as_ptr());
        let bp = vld1q_u32(b_plus[idx..].as_ptr());
        let bm = vld1q_u32(b_minus[idx..].as_ptr());

        // AND operations
        let pp = vandq_u32(ap, bp);
        let mm = vandq_u32(am, bm);
        let pm = vandq_u32(ap, bm);
        let mp = vandq_u32(am, bp);

        // Popcount using NEON's cnt instruction (operates on 8-bit elements)
        // Reinterpret as u8 vectors
        let pp_8 = vreinterpretq_u8_u32(pp);
        let mm_8 = vreinterpretq_u8_u32(mm);
        let pm_8 = vreinterpretq_u8_u32(pm);
        let mp_8 = vreinterpretq_u8_u32(mp);

        // Count bits per byte
        let pp_cnt = vcntq_u8(pp_8);
        let mm_cnt = vcntq_u8(mm_8);
        let pm_cnt = vcntq_u8(pm_8);
        let mp_cnt = vcntq_u8(mp_8);

        // Sum horizontally
        pp_sum += vaddlvq_u8(pp_cnt) as i32;
        mm_sum += vaddlvq_u8(mm_cnt) as i32;
        pm_sum += vaddlvq_u8(pm_cnt) as i32;
        mp_sum += vaddlvq_u8(mp_cnt) as i32;
    }

    // Handle remaining words with scalar code
    let remaining_start = chunks * 4;
    for i in remaining_start..a_plus.len() {
        pp_sum += (a_plus[i] & b_plus[i]).count_ones() as i32;
        mm_sum += (a_minus[i] & b_minus[i]).count_ones() as i32;
        pm_sum += (a_plus[i] & b_minus[i]).count_ones() as i32;
        mp_sum += (a_minus[i] & b_plus[i]).count_ones() as i32;
    }

    pp_sum + mm_sum - pm_sum - mp_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trit::Trit;

    #[test]
    fn test_neon_dot_correctness() {
        let mut a = PackedTritVec::new(256);
        let mut b = PackedTritVec::new(256);

        for i in 0..128 {
            a.set(i, Trit::P);
            b.set(i, if i % 3 == 0 { Trit::P } else { Trit::N });
        }

        let scalar = a.dot(&b);
        let simd = unsafe { dot_neon(&a, &b) };

        assert_eq!(scalar, simd);
    }
}
