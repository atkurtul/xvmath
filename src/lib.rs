#![allow(dead_code)]
#![macro_use]
#![allow(unused_macros)]
//#![cfg(target_feature = "fma")]

#[inline(always)]
const fn shuffle_mask(x : i32, y : i32, z : i32, w : i32) -> i32{
    x | (y << 2) | (z << 4) | (w << 6)
}

macro_rules! shuff4 {
    ($a:expr, $b:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {
        unsafe { vec::new_xmm(_mm_shuffle_ps($a.load(), $b.load(), shuffle_mask($x, $y, $z, $w))) }
    };
}
macro_rules! perm4 {
    ($a:expr,$x:expr, $y:expr, $z:expr, $w:expr) => {
        unsafe { vec::new_xmm(_mm_permute_ps($a.load(), shuffle_mask($x, $y, $z, $w))) }
    };
}

macro_rules! make_swizz {
    ($x:ident, $y:ident, $z:ident, $w:ident, $i:literal, $j:literal, $k:literal, $s:literal) => {
        paste::item! { 
            #[inline(always)]
            pub fn [<$x $y $z $w>](self) -> vec {{ perm4!(self, $i, $j, $k, $s) } } 
            #[inline(always)]
            pub fn [<shuff_ $x $y $z $w>](self, r : vec) -> vec {{ shuff4!(self, r, $i, $j, $k, $s) } }
            #[inline(always)]
            pub fn [<shuff_ $i $j $k $s>](self, r : vec) -> vec {{ shuff4!(self, r, $i, $j, $k, $s) } }
        }
    };
}
macro_rules! make_swizz4 {
    ($x:ident, $y:ident, $z:ident, $i:literal, $j:literal, $k:literal) => {
        make_swizz!($x, $y, $z, x, $i, $j, $k, 0);
        make_swizz!($x, $y, $z, y, $i, $j, $k, 1);
        make_swizz!($x, $y, $z, z, $i, $j, $k, 2);
        make_swizz!($x, $y, $z, w, $i, $j, $k, 3);
    };
}
macro_rules! make_swizz3 {
    ($x:ident, $y:ident, $i:literal, $j:literal) => {
        make_swizz4!($x, $y, x, $i, $j, 0);
        make_swizz4!($x, $y, y, $i, $j, 1);
        make_swizz4!($x, $y, z, $i, $j, 2);
        make_swizz4!($x, $y, w, $i, $j, 3);
    };
}
macro_rules! make_swizz2 {
    ($x:ident, $i:literal) => {
        make_swizz3!($x, x, $i, 0);
        make_swizz3!($x, y, $i, 1);
        make_swizz3!($x, z, $i, 2);
        make_swizz3!($x, w, $i, 3);
    };
}
macro_rules! make_swizz1 {
    () => {
        make_swizz2!(x, 0);
        make_swizz2!(y, 1);
        make_swizz2!(z, 2);
        make_swizz2!(w, 3);
    };
}
macro_rules! def_fn {
    ($name:ident{ $($args:ident),* }) => {
        paste::item! {
            #[inline(always)]
            pub fn [<$name>](self, $($args : vec,)* ) -> vec {
                unsafe { vec::new_xmm([<_mm_ $name _ps>](self.load(), $($args.load(),)*)) }
            }
        }
    }
}

extern crate paste;
extern crate core;
use core::arch::x86_64::*;

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, Default)]
pub struct vec(f32,f32,f32,f32);

#[repr(C, align(32))]
#[derive(Debug, Default, Copy, Clone)]
pub struct mat(vec, vec, vec, vec);

impl vec {
    #[inline(always)]
    pub fn new_xmm(xmm : __m128) -> vec {
        unsafe { *(&xmm as *const __m128 as *const vec) }
    }
    #[inline(always)]
    pub fn load(&self) -> __m128 {
        unsafe { _mm_load_ps(&self.0) }
    }
    #[inline(always)]
    pub fn new(x : f32, y : f32, z : f32, w : f32) -> vec {
        unsafe { vec::new_xmm(_mm_setr_ps(x, y, z, w)) }
    }
    #[inline(always)]
    pub fn dot(self, b : vec) -> vec {
        unsafe { vec::new_xmm(_mm_dp_ps(self.load(), b.load(), 255)) }
    }
    def_fn!(add { b });
    def_fn!(sub { b });
    def_fn!(mul { b });
    def_fn!(div { b });
    def_fn!(xor { b });
    def_fn!(and { b });
    def_fn!(or { b });
    def_fn!(movelh { b });
    def_fn!(movehl { b });
    def_fn!(unpacklo { b });
    def_fn!(unpackhi { b });
    def_fn!(fmadd { b, c });
    def_fn!(fmsub { b, c });
    def_fn!(fnmadd { b, c });
    def_fn!(fnmsub { b, c });
    def_fn!(fmaddsub { b, c });
    make_swizz1!();
}

impl std::ops::Add for vec {
    type Output = vec;
    #[inline(always)]
    fn add(self, r : vec) -> vec { self.add(r) }
}
impl std::ops::Sub for vec {
    type Output = vec;
    #[inline(always)]
    fn sub(self, r : vec) -> vec { self.sub(r) }
}
impl std::ops::Mul for vec {
    type Output = vec;
    #[inline(always)]
    fn mul(self, r : vec) -> vec { self.mul(r) }
}
impl std::ops::Div for vec {
    type Output = vec;
    #[inline(always)]
    fn div(self, r : vec) -> vec { self.div(r) }
}

impl std::ops::Mul<mat> for vec {
    type Output = vec;
    #[inline(always)]
    fn mul(self, m : mat) -> vec { 
        self.xxxx().fmadd(m.0, self.yyyy().fmadd(m.1, self.zzzz().fmadd(m.2, self.wwww() * m.3)))
    }
}

impl std::ops::BitXor for vec {
    type Output = vec;
    #[inline(always)]
    fn bitxor(self, r : vec) -> vec { self.xor(r) }
}

impl std::ops::BitAnd for vec {
    type Output = vec;
    #[inline(always)]
    fn bitand(self, r : vec) -> vec { self.and(r) }
}

impl std::ops::BitOr for vec {
    type Output = vec;
    #[inline(always)]
    fn bitor(self, r : vec) -> vec { self.or(r) }
}

impl mat {
    pub fn rotz(ang : f32) -> mat {
        let c = ang.cos();
        let s = ang.sin();
        let x = vec::new(c, -s, 0., 0.);
        let y = vec::new(s,  c, 0., 0.);
        let z = vec::new(0., 0., 1., 0.);
        let w = vec::new(0., 0., 0., 1.);
        mat(x, y, z, w)
    }
    #[inline(always)]
    pub fn inv(&self) -> mat {
        let m = self;
        let m0 = m.0.movelh(m.1);
        let m1 = m.1.movehl(m.0);
        let m2 = m.2.movelh(m.3);
        let m3 = m.3.movehl(m.2);
        let dc = m3.wwxx().fmsub(m2, m3.yyzz() * m2.zwxy());
        let ab = m0.wwxx().fmsub(m1, m0.yyzz() * m1.zwxy());
        let det = m.0.shuff_xzxz(m.2).fmsub(m.1.shuff_ywyw(m.3), m.0.shuff_ywyw(m.2) * m.1.shuff_xzxz(m.3));
        let xx = det.wwww().fmsub(m0, m1.fmadd(dc.xwxw(), m1.yxwz() * dc.zyzy()));
        let ww = det.xxxx().fmsub(m3, m2.fmadd(ab.xwxw(), m2.yxwz() * ab.zyzy()));
        let yy = det.yyyy().fmsub(m2, m3.fmadd(ab.wxwx(), m3.yxwz() * ab.zyzy()));
        let zz = det.zzzz().fmsub(m1, m0.fmadd(dc.wxwx(), m0.yxwz() * dc.zyzy()));
        let det_m = vec::new(1., -1., -1., 1.) / (det.xxxx().fmadd(det.wwww(), det.yyyy() * det.zzzz()) - ab.dot(dc.xzyw()));
        let xx = xx * det_m;
        let yy = yy * det_m;
        let zz = zz * det_m;
        let ww = ww * det_m;
        mat(xx.shuff_3131(yy), xx.shuff_2020(yy), zz.shuff_3131(ww), zz.shuff_2020(ww))
    }  
    #[inline(always)]
    pub fn tpos(&self) -> mat {
        let m0 = self.0.unpacklo(self.1);
        let m2 = self.2.unpacklo(self.3);
        let m1 = self.0.unpackhi(self.1);
        let m3 = self.2.unpackhi(self.3);
        let x = m0.movelh(m2);
        let y = m2.movehl(m0);
        let z = m1.movelh(m3);
        let w = m3.movehl(m1);
        mat(x, y, z, w)
    }

}

impl std::ops::Add for mat {
    type Output = mat;
    #[inline(always)]
    fn add(self, r : mat) -> mat {  mat(self.0 + r.1, self.1 + r.1, self.2 + r.2, self.3 + r.3) }
}
impl std::ops::Sub for mat {
    type Output = mat;
    #[inline(always)]
    fn sub(self, r : mat) -> mat {  mat(self.0 - r.1, self.1 - r.1, self.2 - r.2, self.3 - r.3) }
}

impl std::ops::Mul for mat {
    type Output = mat;
    #[inline(always)]
    fn mul(self, r : mat) -> mat {
        mat(self.0 * r, self.1 * r, self.2 * r, self.3 * r)
    }
}

impl std::ops::Mul<vec> for mat {
    type Output = vec;
    #[inline(always)]
    fn mul(self, r : vec) -> vec {
        r * self.tpos()
    }
}
