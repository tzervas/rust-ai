//! Basic usage of ternary-rs.
//!
//! Run with: `cargo run --example basic`

use trit_vsa::{PackedTritVec, Trit, Tryte3, Word6};

fn main() {
    println!("=== trit-vsa Basic Example ===\n");

    // Trit operations
    println!("1. Trit Operations");
    let a = Trit::P; // +1
    let b = Trit::N; // -1
    let c = Trit::Z; // 0

    println!("   a = {} (value: {})", a, a.value());
    println!("   b = {} (value: {})", b, b.value());
    println!("   c = {} (value: {})", c, c.value());
    println!("   a * b = {} (should be -1)", a * b);
    println!("   -a = {} (should be -1)", -a);

    // Tryte3 - 3 trits
    println!("\n2. Tryte3 (3 trits, range -13 to +13)");
    let x = Tryte3::from_value(7).unwrap();
    let y = Tryte3::from_value(5).unwrap();
    println!("   x = {:?}", x);
    println!("   y = {:?}", y);

    let (sum, carry) = x + y;
    let total = sum.value() + carry.value() as i32 * 27;
    println!(
        "   x + y = {} (sum={}, carry={})",
        total,
        sum.value(),
        carry.value()
    );

    // Word6 - 6 trits
    println!("\n3. Word6 (6 trits, range -364 to +364)");
    let w1 = Word6::from_value(100).unwrap();
    let w2 = Word6::from_value(200).unwrap();
    println!("   w1 = {:?}", w1);
    println!("   w2 = {:?}", w2);

    let (sum, carry) = w1 + w2;
    let total = sum.value() + carry.value() as i32 * 729;
    println!(
        "   w1 + w2 = {} (sum={}, carry={})",
        total,
        sum.value(),
        carry.value()
    );

    // PackedTritVec - high-dimensional vectors
    println!("\n4. PackedTritVec (bitsliced storage)");
    let mut vec1 = PackedTritVec::new(1000);
    let mut vec2 = PackedTritVec::new(1000);

    // Set some values
    for i in 0..500 {
        vec1.set(i, Trit::P);
        vec2.set(i, if i % 2 == 0 { Trit::P } else { Trit::N });
    }

    println!("   vec1: {} non-zeros out of 1000", vec1.count_nonzero());
    println!("   vec2: {} non-zeros out of 1000", vec2.count_nonzero());
    println!("   vec1.dot(vec2) = {}", vec1.dot(&vec2));
    println!("   vec1.sum() = {}", vec1.sum());

    println!("\nDone!");
}
