//! Vector Symbolic Architecture (VSA) operations.
//!
//! Run with: `cargo run --example vsa`

use trit_vsa::{vsa, PackedTritVec, Trit};

fn main() {
    println!("=== trit-vsa VSA Example ===\n");

    let dim = 10000; // High-dimensional vectors for VSA

    // Create random-like symbol vectors
    println!("1. Creating symbol vectors (dim={})", dim);
    let dog = create_random_vector(dim, 42);
    let cat = create_random_vector(dim, 123);
    let animal = create_random_vector(dim, 456);
    let pet = create_random_vector(dim, 789);

    println!("   Created: dog, cat, animal, pet");

    // Bundle: superposition (dog + cat = pets)
    println!("\n2. Bundle operation (superposition)");
    let pets = vsa::bundle(&dog, &cat);
    println!("   pets = bundle(dog, cat)");
    println!(
        "   similarity(pets, dog) = {:.3}",
        vsa::cosine_similarity(&pets, &dog)
    );
    println!(
        "   similarity(pets, cat) = {:.3}",
        vsa::cosine_similarity(&pets, &cat)
    );
    println!(
        "   similarity(pets, animal) = {:.3}",
        vsa::cosine_similarity(&pets, &animal)
    );

    // Bind: create associations
    println!("\n3. Bind operation (association)");
    let dog_is_animal = vsa::bind(&dog, &animal);
    let cat_is_pet = vsa::bind(&cat, &pet);
    println!("   dog_is_animal = bind(dog, animal)");
    println!("   cat_is_pet = bind(cat, pet)");

    // Query: what is dog?
    println!("\n4. Query: What is dog? (unbind dog_is_animal with dog)");
    let query_result = vsa::unbind(&dog_is_animal, &dog);
    println!(
        "   similarity(result, animal) = {:.3}",
        vsa::cosine_similarity(&query_result, &animal)
    );
    println!(
        "   similarity(result, pet) = {:.3}",
        vsa::cosine_similarity(&query_result, &pet)
    );
    println!(
        "   similarity(result, cat) = {:.3}",
        vsa::cosine_similarity(&query_result, &cat)
    );

    // Bundle multiple associations
    println!("\n5. Complex structure: bundle of bindings");
    let knowledge = vsa::bundle(&dog_is_animal, &cat_is_pet);
    println!("   knowledge = bundle(dog_is_animal, cat_is_pet)");

    // Can recover both facts
    let recovered_animal = vsa::unbind(&knowledge, &dog);
    let recovered_pet = vsa::unbind(&knowledge, &cat);
    println!(
        "   unbind(knowledge, dog) ~ animal: {:.3}",
        vsa::cosine_similarity(&recovered_animal, &animal)
    );
    println!(
        "   unbind(knowledge, cat) ~ pet: {:.3}",
        vsa::cosine_similarity(&recovered_pet, &pet)
    );

    // Hamming distance
    println!("\n6. Distance metrics");
    println!(
        "   hamming(dog, cat) = {}",
        vsa::hamming_distance(&dog, &cat)
    );
    println!(
        "   hamming(dog, dog) = {}",
        vsa::hamming_distance(&dog, &dog)
    );

    println!("\nDone!");
}

/// Create a pseudo-random ternary vector.
fn create_random_vector(dim: usize, seed: u64) -> PackedTritVec {
    let mut vec = PackedTritVec::new(dim);
    let mut state = seed;

    for i in 0..dim {
        // Simple LCG for pseudo-random numbers
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (state >> 62) as i8 - 1; // -1, 0, or 1

        let trit = match val {
            -1 => Trit::N,
            0 => Trit::Z,
            _ => Trit::P,
        };
        vec.set(i, trit);
    }

    vec
}
