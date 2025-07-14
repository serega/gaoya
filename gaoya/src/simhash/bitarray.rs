//! Fixed-size bit array. `LANES` = number of 64-bit words.
//! Total bits = `LANES * 64`.  Works on stable Rust.

use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::*,
};
use num_traits::{One, Zero};

#[derive(Clone, Copy)]
pub struct BitArray<const LANES: usize> {
    lanes: [u64; LANES],
}
impl<const LANES: usize> Default for BitArray<LANES> {
    fn default() -> Self {
        Self { lanes: [0u64; LANES] }
    }
}

// simple traits 

impl<const L: usize> fmt::Debug for BitArray<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitArray<{}×64>(…)", L)
    }
}
impl<const L: usize> PartialEq for BitArray<L> {
    fn eq(&self, o: &Self) -> bool { self.lanes == o.lanes }
}
impl<const L: usize> Eq for BitArray<L> {}
impl<const L: usize> PartialOrd for BitArray<L> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.lanes.cmp(&o.lanes)) }
}
impl<const L: usize> Hash for BitArray<L> {
    fn hash<H: Hasher>(&self, st: &mut H) { for &x in &self.lanes { st.write_u64(x) } }
}

/* ---------- Zero / One (need Add / Mul dummies) -------------------------- */

impl<const L: usize> Add for BitArray<L> { type Output = Self; fn add(self,_:Self)->Self{self} }
impl<const L: usize> Mul for BitArray<L> { type Output = Self; fn mul(self,_:Self)->Self{self} }

impl<const L: usize> Zero for BitArray<L> {
    fn zero() -> Self { Self { lanes: [0; L] } }
    fn is_zero(&self) -> bool { self.lanes.iter().all(|&x| x == 0) }
}
impl<const L: usize> One for BitArray<L> {
    fn one() -> Self {
        let mut a = [0u64; L];
        a[0] = 1;
        Self { lanes: a }
    }
}

/* ---------- bitwise ops -------------------------------------------------- */

macro_rules! impl_bin {
    ($Trait:ident,$func:ident,$op:tt) => {
        impl<const L: usize> $Trait for BitArray<L>{
            type Output = Self;
            fn $func(mut self, rhs:Self)->Self{
                for (a,b) in self.lanes.iter_mut().zip(rhs.lanes){ *a = *a $op b }
                self
            }
        }
        impl<const L: usize> $Trait<&Self> for BitArray<L>{
            type Output = Self;
            fn $func(self, rhs:&Self)->Self{ self.$func(*rhs) }
        }
    };
}
impl_bin!(BitOr,  bitor , |);
impl_bin!(BitAnd, bitand, &);
impl_bin!(BitXor, bitxor, ^);

impl<const L: usize> Not for BitArray<L>{
    type Output=Self;
    fn not(mut self)->Self{ for x in &mut self.lanes{*x= !*x} self }
}
impl<const L: usize> BitOrAssign for BitArray<L>{
    fn bitor_assign(&mut self, rhs:Self){ for (a,b) in self.lanes.iter_mut().zip(rhs.lanes){*a|=b;} }
}

/* ---------- shifts ------------------------------------------------------- */

impl<const L: usize> ShlAssign<usize> for BitArray<L>{
    fn shl_assign(&mut self, n: usize){
        let words = n / 64; let bits = n % 64;
        if words>0 {
            for i in (0..L).rev() {
                self.lanes[i] = if i>=words { self.lanes[i-words] } else {0};
            }
        }
        if bits>0 {
            for i in (0..L).rev() {
                self.lanes[i] <<= bits;
                if i>0 { self.lanes[i] |= self.lanes[i-1] >> (64-bits); }
            }
        }
    }
}
impl<const L: usize> Shl<usize> for BitArray<L>{ type Output=Self; fn shl(mut self,n:usize)->Self{self<<=n;self}}

impl<const L: usize> ShrAssign<usize> for BitArray<L>{
    fn shr_assign(&mut self, n: usize){
        let words=n/64; let bits=n%64;
        if words>0 {
            for i in 0..L {
                self.lanes[i] = if i+words<L { self.lanes[i+words] } else {0};
            }
        }
        if bits>0 {
            for i in 0..L {
                self.lanes[i] >>= bits;
                if i+1<L { self.lanes[i]|= self.lanes[i+1] << (64-bits);}
            }
        }
    }
}
impl<const L: usize> Shr<usize> for BitArray<L>{ type Output=Self; fn shr(mut self,n:usize)->Self{self>>=n;self}}

/* ---------- SimHashBits -------------------------------------------------- */

impl<const L: usize> crate::simhash::SimHashBits for BitArray<L>{
    fn count_ones(self)->usize{ self.lanes.iter().map(|x|x.count_ones() as usize).sum() }
    fn to_u32_high_bits(self)->u32{ (self.lanes[L-1]>>32) as u32 }
    fn to_u64_high_bits(self)->u64{ self.lanes[L-1] }
    fn hamming_distance(&self,r:&Self)->usize{
        self.lanes.iter().zip(r.lanes)
            .map(|(a,b)| (a^b).count_ones() as usize).sum()
    }
    fn bit_length()->usize{ L*64 }
}