//===-- APInt.cpp - Implement APInt class ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Neil Booth and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a variety of operations on a representation of
// arbitrary precision, two's-complement, bignum integer values.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include "APFloat.h"

using namespace llvm;

// This implements a variety of operations on a representation of
// arbitrary precision, two's-complement, bignum integer values.

/* Assumed by lowHalf, highHalf, partMSB and partLSB.  A fairly safe
   and unrestricting assumption.  */
COMPILE_TIME_ASSERT(integerPartWidth % 2 == 0);

/* Some handy functions local to this file.  */
namespace {

  /* Returns the integer part with the least significant BITS set.
     BITS cannot be zero.  */
  inline integerPart
  lowBitMask(unsigned int bits)
  {
    assert (bits != 0 && bits <= integerPartWidth);

    return ~(integerPart) 0 >> (integerPartWidth - bits);
  }

  /* Returns the value of the lower nibble of PART.  */
  inline integerPart
  lowHalf(integerPart part)
  {
    return part & lowBitMask(integerPartWidth / 2);
  }

  /* Returns the value of the upper nibble of PART.  */
  inline integerPart
  highHalf(integerPart part)
  {
    return part >> (integerPartWidth / 2);
  }

  /* Returns the bit number of the most significant bit of a part.  If
     the input number has no bits set -1U is returned.  */
  unsigned int
  partMSB(integerPart value)
  {
    unsigned int n, msb;

    if (value == 0)
      return -1U;

    n = integerPartWidth / 2;

    msb = 0;
    do {
      if (value >> n) {
        value >>= n;
        msb += n;
      }

      n >>= 1;
    } while (n);

    return msb;
  }

  /* Returns the bit number of the least significant bit of a part.
     If the input number has no bits set -1U is returned.  */
  unsigned int
  partLSB(integerPart value)
  {
    unsigned int n, lsb;

    if (value == 0)
      return -1U;

    lsb = integerPartWidth - 1;
    n = integerPartWidth / 2;

    do {
      if (value << n) {
        value <<= n;
        lsb -= n;
      }

      n >>= 1;
    } while (n);

    return lsb;
  }
}

/* Sets the least significant part of a bignum to the input value, and
   zeroes out higher parts.  */
void
APInt::tcSet(integerPart *dst, integerPart part, unsigned int parts)
{
  unsigned int i;

  dst[0] = part;
  for(i = 1; i < parts; i++)
    dst[i] = 0;
}

/* Assign one bignum to another.  */
void
APInt::tcAssign(integerPart *dst, const integerPart *src, unsigned int parts)
{
  unsigned int i;

  for(i = 0; i < parts; i++)
    dst[i] = src[i];
}

/* Returns true if a bignum is zero, false otherwise.  */
bool
APInt::tcIsZero(const integerPart *src, unsigned int parts)
{
  unsigned int i;

  for(i = 0; i < parts; i++)
    if (src[i])
      return false;

  return true;
}

/* Extract the given bit of a bignum; returns 0 or 1.  */
int
APInt::tcExtractBit(const integerPart *parts, unsigned int bit)
{
  return(parts[bit / integerPartWidth]
         & ((integerPart) 1 << bit % integerPartWidth)) != 0;
}

/* Set the given bit of a bignum.  */
void
APInt::tcSetBit(integerPart *parts, unsigned int bit)
{
  parts[bit / integerPartWidth] |= (integerPart) 1 << (bit % integerPartWidth);
}

/* Returns the bit number of the least significant bit of a number.
   If the input number has no bits set -1U is returned.  */
unsigned int
APInt::tcLSB(const integerPart *parts, unsigned int n)
{
  unsigned int i, lsb;

  for(i = 0; i < n; i++) {
      if (parts[i] != 0) {
          lsb = partLSB(parts[i]);

          return lsb + i * integerPartWidth;
      }
  }

  return -1U;
}

/* Returns the bit number of the most significant bit of a number.  If
   the input number has no bits set -1U is returned.  */
unsigned int
APInt::tcMSB(const integerPart *parts, unsigned int n)
{
  unsigned int msb;

  do {
      --n;

      if (parts[n] != 0) {
          msb = partMSB(parts[n]);

          return msb + n * integerPartWidth;
      }
  } while (n);

  return -1U;
}

/* DST += RHS + C where C is zero or one.  Returns the carry flag.  */
integerPart
APInt::tcAdd(integerPart *dst, const integerPart *rhs,
             integerPart c, unsigned int parts)
{
  unsigned int i;

  assert(c <= 1);

  for(i = 0; i < parts; i++) {
    integerPart l;

    l = dst[i];
    if (c) {
      dst[i] += rhs[i] + 1;
      c = (dst[i] <= l);
    } else {
      dst[i] += rhs[i];
      c = (dst[i] < l);
    }
  }

  return c;
}

/* DST -= RHS + C where C is zero or one.  Returns the carry flag.  */
integerPart
APInt::tcSubtract(integerPart *dst, const integerPart *rhs,
                  integerPart c, unsigned int parts)
{
  unsigned int i;

  assert(c <= 1);

  for(i = 0; i < parts; i++) {
    integerPart l;

    l = dst[i];
    if (c) {
      dst[i] -= rhs[i] + 1;
      c = (dst[i] >= l);
    } else {
      dst[i] -= rhs[i];
      c = (dst[i] > l);
    }
  }

  return c;
}

/* Negate a bignum in-place.  */
void
APInt::tcNegate(integerPart *dst, unsigned int parts)
{
  tcComplement(dst, parts);
  tcIncrement(dst, parts);
}

/*  DST += SRC * MULTIPLIER + PART   if add is true
    DST  = SRC * MULTIPLIER + PART   if add is false

    Requires 0 <= DSTPARTS <= SRCPARTS + 1.  If DST overlaps SRC
    they must start at the same point, i.e. DST == SRC.

    If DSTPARTS == SRCPARTS + 1 no overflow occurs and zero is
    returned.  Otherwise DST is filled with the least significant
    DSTPARTS parts of the result, and if all of the omitted higher
    parts were zero return zero, otherwise overflow occurred and
    return one.  */
int
APInt::tcMultiplyPart(integerPart *dst, const integerPart *src,
                      integerPart multiplier, integerPart carry,
                      unsigned int srcParts, unsigned int dstParts,
                      bool add)
{
  unsigned int i, n;

  /* Otherwise our writes of DST kill our later reads of SRC.  */
  assert(dst <= src || dst >= src + srcParts);
  assert(dstParts <= srcParts + 1);

  /* N loops; minimum of dstParts and srcParts.  */
  n = dstParts < srcParts ? dstParts: srcParts;

  for(i = 0; i < n; i++) {
    integerPart low, mid, high, srcPart;

      /* [ LOW, HIGH ] = MULTIPLIER * SRC[i] + DST[i] + CARRY.

         This cannot overflow, because

         (n - 1) * (n - 1) + 2 (n - 1) = (n - 1) * (n + 1)

         which is less than n^2.  */

    srcPart = src[i];

    if (multiplier == 0 || srcPart == 0)        {
      low = carry;
      high = 0;
    } else {
      low = lowHalf(srcPart) * lowHalf(multiplier);
      high = highHalf(srcPart) * highHalf(multiplier);

      mid = lowHalf(srcPart) * highHalf(multiplier);
      high += highHalf(mid);
      mid <<= integerPartWidth / 2;
      if (low + mid < low)
        high++;
      low += mid;

      mid = highHalf(srcPart) * lowHalf(multiplier);
      high += highHalf(mid);
      mid <<= integerPartWidth / 2;
      if (low + mid < low)
        high++;
      low += mid;

      /* Now add carry.  */
      if (low + carry < low)
        high++;
      low += carry;
    }

    if (add) {
      /* And now DST[i], and store the new low part there.  */
      if (low + dst[i] < low)
        high++;
      dst[i] += low;
    } else
      dst[i] = low;

    carry = high;
  }

  if (i < dstParts) {
    /* Full multiplication, there is no overflow.  */
    assert(i + 1 == dstParts);
    dst[i] = carry;
    return 0;
  } else {
    /* We overflowed if there is carry.  */
    if (carry)
      return 1;

    /* We would overflow if any significant unwritten parts would be
       non-zero.  This is true if any remaining src parts are non-zero
       and the multiplier is non-zero.  */
    if (multiplier)
      for(; i < srcParts; i++)
        if (src[i])
          return 1;

    /* We fitted in the narrow destination.  */
    return 0;
  }
}

/* DST = LHS * RHS, where DST has the same width as the operands and
   is filled with the least significant parts of the result.  Returns
   one if overflow occurred, otherwise zero.  DST must be disjoint
   from both operands.  */
int
APInt::tcMultiply(integerPart *dst, const integerPart *lhs,
                  const integerPart *rhs, unsigned int parts)
{
  unsigned int i;
  int overflow;

  assert(dst != lhs && dst != rhs);

  overflow = 0;
  tcSet(dst, 0, parts);

  for(i = 0; i < parts; i++)
    overflow |= tcMultiplyPart(&dst[i], lhs, rhs[i], 0, parts,
                               parts - i, true);

  return overflow;
}

/* DST = LHS * RHS, where DST has twice the width as the operands.  No
   overflow occurs.  DST must be disjoint from both operands.  */
void
APInt::tcFullMultiply(integerPart *dst, const integerPart *lhs,
                      const integerPart *rhs, unsigned int parts)
{
  unsigned int i;
  int overflow;

  assert(dst != lhs && dst != rhs);

  overflow = 0;
  tcSet(dst, 0, parts);

  for(i = 0; i < parts; i++)
    overflow |= tcMultiplyPart(&dst[i], lhs, rhs[i], 0, parts,
                               parts + 1, true);

  assert(!overflow);
}

/* If RHS is zero LHS and REMAINDER are left unchanged, return one.
   Otherwise set LHS to LHS / RHS with the fractional part discarded,
   set REMAINDER to the remainder, return zero.  i.e.

   OLD_LHS = RHS * LHS + REMAINDER

   SCRATCH is a bignum of the same size as the operands and result for
   use by the routine; its contents need not be initialized and are
   destroyed.  LHS, REMAINDER and SCRATCH must be distinct.
*/
int
APInt::tcDivide(integerPart *lhs, const integerPart *rhs,
                integerPart *remainder, integerPart *srhs,
                unsigned int parts)
{
  unsigned int n, shiftCount;
  integerPart mask;

  assert(lhs != remainder && lhs != srhs && remainder != srhs);

  shiftCount = tcMSB(rhs, parts) + 1;
  if (shiftCount == 0)
    return true;

  shiftCount = parts * integerPartWidth - shiftCount;
  n = shiftCount / integerPartWidth;
  mask = (integerPart) 1 << (shiftCount % integerPartWidth);

  tcAssign(srhs, rhs, parts);
  tcShiftLeft(srhs, parts, shiftCount);
  tcAssign(remainder, lhs, parts);
  tcSet(lhs, 0, parts);

  /* Loop, subtracting SRHS if REMAINDER is greater and adding that to
     the total.  */
  for(;;) {
      int compare;

      compare = tcCompare(remainder, srhs, parts);
      if (compare >= 0) {
        tcSubtract(remainder, srhs, 0, parts);
        lhs[n] |= mask;
      }

      if (shiftCount == 0)
        break;
      shiftCount--;
      tcShiftRight(srhs, parts, 1);
      if ((mask >>= 1) == 0)
        mask = (integerPart) 1 << (integerPartWidth - 1), n--;
  }

  return false;
}

/* Shift a bignum left COUNT bits in-place.  Shifted in bits are zero.
   There are no restrictions on COUNT.  */
void
APInt::tcShiftLeft(integerPart *dst, unsigned int parts, unsigned int count)
{
  unsigned int jump, shift;

  /* Jump is the inter-part jump; shift is is intra-part shift.  */
  jump = count / integerPartWidth;
  shift = count % integerPartWidth;

  while (parts > jump) {
    integerPart part;

    parts--;

    /* dst[i] comes from the two parts src[i - jump] and, if we have
       an intra-part shift, src[i - jump - 1].  */
    part = dst[parts - jump];
    if (shift) {
      part <<= shift;
        if (parts >= jump + 1)
          part |= dst[parts - jump - 1] >> (integerPartWidth - shift);
      }

    dst[parts] = part;
  }

  while (parts > 0)
    dst[--parts] = 0;
}

/* Shift a bignum right COUNT bits in-place.  Shifted in bits are
   zero.  There are no restrictions on COUNT.  */
void
APInt::tcShiftRight(integerPart *dst, unsigned int parts, unsigned int count)
{
  unsigned int i, jump, shift;

  /* Jump is the inter-part jump; shift is is intra-part shift.  */
  jump = count / integerPartWidth;
  shift = count % integerPartWidth;

  /* Perform the shift.  This leaves the most significant COUNT bits
     of the result at zero.  */
  for(i = 0; i < parts; i++) {
    integerPart part;

    if (i + jump >= parts) {
      part = 0;
    } else {
      part = dst[i + jump];
      if (shift) {
        part >>= shift;
        if (i + jump + 1 < parts)
          part |= dst[i + jump + 1] << (integerPartWidth - shift);
      }
    }

    dst[i] = part;
  }
}

/* Bitwise and of two bignums.  */
void
APInt::tcAnd(integerPart *dst, const integerPart *rhs, unsigned int parts)
{
  unsigned int i;

  for(i = 0; i < parts; i++)
    dst[i] &= rhs[i];
}

/* Bitwise inclusive or of two bignums.  */
void
APInt::tcOr(integerPart *dst, const integerPart *rhs, unsigned int parts)
{
  unsigned int i;

  for(i = 0; i < parts; i++)
    dst[i] |= rhs[i];
}

/* Bitwise exclusive or of two bignums.  */
void
APInt::tcXor(integerPart *dst, const integerPart *rhs, unsigned int parts)
{
  unsigned int i;

  for(i = 0; i < parts; i++)
    dst[i] ^= rhs[i];
}

/* Complement a bignum in-place.  */
void
APInt::tcComplement(integerPart *dst, unsigned int parts)
{
  unsigned int i;

  for(i = 0; i < parts; i++)
    dst[i] = ~dst[i];
}

/* Comparison (unsigned) of two bignums.  */
int
APInt::tcCompare(const integerPart *lhs, const integerPart *rhs,
                 unsigned int parts)
{
  while (parts) {
      parts--;
      if (lhs[parts] == rhs[parts])
        continue;

      if (lhs[parts] > rhs[parts])
        return 1;
      else
        return -1;
    }

  return 0;
}

/* Increment a bignum in-place, return the carry flag.  */
integerPart
APInt::tcIncrement(integerPart *dst, unsigned int parts)
{
  unsigned int i;

  for(i = 0; i < parts; i++)
    if (++dst[i] != 0)
      break;

  return i == parts;
}

/* Set the least significant BITS bits of a bignum, clear the
   rest.  */
void
APInt::tcSetLeastSignificantBits(integerPart *dst, unsigned int parts,
                                 unsigned int bits)
{
  unsigned int i;

  i = 0;
  while (bits > integerPartWidth) {
    dst[i++] = ~(integerPart) 0;
    bits -= integerPartWidth;
  }

  if (bits)
    dst[i++] = ~(integerPart) 0 >> (integerPartWidth - bits);

  while (i < parts)
    dst[i++] = 0;
}
