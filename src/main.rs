extern crate xvmath;
use xvmath::*;


fn debranch(a : vec, b : vec, c : vec) -> vec {
    (c & a) | c.andnot(b)
}

fn main() {
    let a = vec::newss(3.);
    let b = vec::newss(2.);
    println!("{:?}", debranch(a, b, a.lt(b)));
}