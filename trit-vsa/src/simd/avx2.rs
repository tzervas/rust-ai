//! AVX2 SIMD implementations for x86_64.

#![cfg(all(feature = "simd", target_arch = "x86_64"))]

use crate::packed::PackedTritVec;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-accelerated dot product.
///
/// # Safety
///
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2")]
pub unsafe fn dot_avx2(a: &PackedTritVec, b: &PackedTritVec) -> i32 {
    let a_plus = a.plus_plane();
    let a_minus = a.minus_plane();
    let b_plus = b.plus_plane();
    let b_minus = b.minus_plane();

    let mut pp_sum: i32 = 0;
    let mut mm_sum: i32 = 0;
    let mut pm_sum: i32 = 0;
    let mut mp_sum: i32 = 0;

    // Process 8 u32s (256 bits) at a time
    let chunks = a_plus.len() / 8;

    for i in 0..chunks {
        let idx = i * 8;

        // Load 256 bits from each plane
        let ap = _mm256_loadu_si256(a_plus[idx..].as_ptr().cast());
        let am = _mm256_loadu_si256(a_minus[idx..].as_ptr().cast());
        let bp = _mm256_loadu_si256(b_plus[idx..].as_ptr().cast());
        let bm = _mm256_loadu_si256(b_minus[idx..].as_ptr().cast());

        // AND operations
        let pp = _mm256_and_si256(ap, bp);
        let mm = _mm256_and_si256(am, bm);
        let pm = _mm256_and_si256(ap, bm);
        let mp = _mm256_and_si256(am, bp);

        // Popcount via horizontal add (AVX2 doesn't have direct popcount for 256-bit)
        // We'll extract and use scalar popcount for now
        let pp_arr: [u64; 4] = std::mem::transmute(pp);
        let mm_arr: [u64; 4] = std::mem::transmute(mm);
        let pm_arr: [u64; 4] = std::mem::transmute(pm);
        let mp_arr: [u64; 4] = std::mem::transmute(mp);

        for j in 0..4 {
            pp_sum += pp_arr[j].count_ones() as i32;
            mm_sum += mm_arr[j].count_ones() as i32;
            pm_sum += pm_arr[j].count_ones() as i32;
            mp_sum += mp_arr[j].count_ones() as i32;
        }
    }

    // Handle remaining words with scalar code
    let remaining_start = chunks * 8;
    for i in remaining_start..a_plus.len() {
        pp_sum += (a_plus[i] & b_plus[i]).count_ones() as i32;
        mm_sum += (a_minus[i] & b_minus[i]).count_ones() as i32;
        pm_sum += (a_plus[i] & b_minus[i]).count_ones() as i32;
        mp_sum += (a_minus[i] & b_plus[i]).count_ones() as i32;
    }

    pp_sum + mm_sum - pm_sum - mp_sum
}

/// AVX2-accelerated total popcount.
///
/// # Safety
///
/// Caller must ensure POPCNT is available.
#[target_feature(enable = "popcnt")]
pub unsafe fn popcount_total_avx2(vec: &PackedTritVec) -> usize {
    let plus = vec.plus_plane();
    let minus = vec.minus_plane();

    let mut total: u64 = 0;

    for i in 0..plus.len() {
        total += _popcnt64(plus[i] as i64) as u64;
        total += _popcnt64(minus[i] as i64) as u64;
    }

    total as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trit::Trit;

    #[test]
    fn test_avx2_dot_correctness() {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip on non-AVX2 systems
        }

        let mut a = PackedTritVec::new(512);
        let mut b = PackedTritVec::new(512);

        for i in 0..256 {
            a.set(i, Trit::P);
            b.set(i, if i % 3 == 0 { Trit::P } else { Trit::N });
        }

        let scalar = a.dot(&b);
        let simd = unsafe { dot_avx2(&a, &b) };

        assert_eq!(scalar, simd);
    }
}
