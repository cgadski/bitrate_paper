use rand::Rng;

const MAX_D: usize = 1 << 13;
const D_STEP: usize = 164;
const MAX_N: usize = 1 << 16;
const MAX_K: usize = 101;
const K_STEP: usize = 2;

fn new_random(rng: &mut impl Rng) -> Vec<i16> {
    (0..MAX_D)
        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
        .collect()
}

fn randomize(x: &mut [i16], rng: &mut impl Rng) {
    let mut i = 0;
    while i < x.len() {
        let bits = rng.gen::<u64>();
        for j in 0..64.min(x.len() - i) {
            x[i + j] = if (bits >> j) & 1 == 0 { -1 } else { 1 };
        }
        i += 64;
    }
}

// Given k, produce (correlated) samples of the events threshold(k, N, d), topk(k, N, d)
// for various values of N and d.
fn sample_k(k: usize) {
    let mut rng = rand::thread_rng();

    let codes: Vec<Vec<i16>> = (0..k).map(|_| new_random(&mut rng)).collect();
    let mut code: Vec<i16> = vec![0; MAX_D];

    for c in 0..k {
        for d in 0..MAX_D {
            code[d] += codes[c][d];
        }
    }

    let mut threshold_errors = [0 as u32; MAX_D];
    let mut topk_errors = [0 as u32; MAX_D];

    // Look at the code words for each symbol
    let mut min_inner = [i16::MAX; MAX_D];

    for c in 0..k {
        let this_code = &codes[c];
        let mut inner = 0;
        for d in 0..MAX_D {
            inner += this_code[d] * code[d];
            if inner < (d as i16) / 2 {
                threshold_errors[d] += 1;
            }
            if inner < min_inner[d] {
                min_inner[d] = inner;
            }
        }
    }

    // Construct new codes and check if they would interfere with either threshold
    // or topk decoding of our code.
    let mut new_code: [i16; MAX_D] = [0; MAX_D];
    let mut n = k;
    loop {
        randomize(&mut new_code, &mut rng);

        n += 1;
        let good_n = n & (n - 1) == 0 && n >= 256;
        if good_n {
            // eprintln!("  n = {}", n);
        }

        let mut inner = 0;
        for d in 0..MAX_D {
            inner += new_code[d] * code[d];
            if inner >= (d as i16) / 2 {
                threshold_errors[d] += 1;
            }
            if inner >= min_inner[d] {
                topk_errors[d] += 1;
            }
            if good_n && d > 0 && (d % D_STEP == 0) {
                println!("threshold, {}, {}, {}, {}", k, n, d, threshold_errors[d]);
                println!("topk, {}, {}, {}, {}", k, n, d, topk_errors[d]);
            }
        }

        if n == MAX_N {
            break;
        }
    }
}

fn main() {
    let mut k = 2;
    loop {
        eprintln!("k = {} / 100", k);
        sample_k(k);
        k += K_STEP;
        if k >= MAX_K {
            break;
        }
    }
}
