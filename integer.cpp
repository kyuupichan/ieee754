/*
   Copyright 2005-2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.

   Two's complement bignum arithmetic.  The source code will run
   correctly on any host - sign magnitude, one's complement or two's
   complement.

   The code is reasonably efficient given the above constraint.
*/

#include <cassert>
#include "float.h"

using namespace llvm;

/* Assumed by low_half, high_half, part_msb and part_lsb.  A fairly
   safe and unrestricting assumption.  */
compile_time_assert (t_integer_part_width % 2 == 0);

#define low_half(part) ((part) & low_bit_mask (t_integer_part_width / 2))
#define high_half(part) ((part) >> (t_integer_part_width / 2))
#define low_bit_mask(bits) \
	(~(t_integer_part) 0 >> (t_integer_part_width - (bits)))

/* Some handy functions local to this file.  */
namespace {

/* Returns the bit number of the most significant bit of a part.
   If the input number has no bits set zero is returned.  */
unsigned int
tc_part_msb (t_integer_part value)
{
  unsigned int n, msb;

  if (value == 0)
    return 0;

  n = t_integer_part_width / 2;

  msb = 1;
  do
    {
      if (value >> n)
	{
	  value >>= n;
	  msb += n;
	}

      n >>= 1;
    }
  while (n);

  return msb;
}

/* Returns the bit number of the least significant bit of a part.
   If the input number has no bits set zero is returned.  */
unsigned int
part_lsb (t_integer_part value)
{
  unsigned int n, lsb;

  if (value == 0)
    return 0;

  lsb = t_integer_part_width;
  n = t_integer_part_width / 2;

  do
    {
      if (value << n)
	{
	  value <<= n;
	  lsb -= n;
	}

      n >>= 1;
    }
  while (n);

  return lsb;
}
}

/* Sets the least significant part of a bignum to the input value, and
   zeroes out higher parts.  */
void
APInt::tc_set (t_integer_part *dst, t_integer_part part, unsigned int parts)
{
  unsigned int i;

  dst[0] = part;
  for (i = 1; i < parts; i++)
    dst[i] = 0;
}

/* Assign one bignum to another.  */
void
APInt::tc_assign (t_integer_part *dst, const t_integer_part *src,
		 unsigned int parts)
{
  unsigned int i;

  for (i = 0; i < parts; i++)
    dst[i] = src[i];
}

/* Returns true if a bignum is zero, false otherwise.  */
bool
APInt::tc_is_zero (const t_integer_part *src, unsigned int parts)
{
  unsigned int i;

  for (i = 0; i < parts; i++)
    if (src[i])
      return false;

  return true;
}

/* Extract the given bit of a bignum; returns 0 or 1.  BIT cannot be
   zero.  */
int
APInt::tc_extract_bit (const t_integer_part *parts, unsigned int bit)
{
  assert (bit != 0);

  bit--;

  return (parts[bit / t_integer_part_width]
	  & ((t_integer_part) 1 << bit % t_integer_part_width)) != 0;
}

/* Set the given bit of a bignum.  BIT cannot be zero.  */
void
APInt::tc_set_bit (t_integer_part *parts, unsigned int bit)
{
  assert (bit != 0);

  bit--;

  parts[bit / t_integer_part_width]
    |= (t_integer_part) 1 << (bit % t_integer_part_width);
}

/* Returns the bit number of the least significant bit of a number.
   If the input number has no bits set zero is returned.  */
unsigned int
APInt::tc_lsb (const t_integer_part *parts, unsigned int n)
{
  unsigned int i, lsb;

  for (i = 0; i < n; i++)
    {
      if (parts[i] != 0)
	{
	  lsb = part_lsb (parts[i]);

	  return lsb + i * t_integer_part_width;
	}
    }

  return 0;
}

/* Returns the bit number of the most significant bit of a number.  If
   the input number has no bits set zero is returned.  */
unsigned int
APInt::tc_msb (const t_integer_part *parts, unsigned int n)
{
  unsigned int msb;

  do
    {
      --n;

      if (parts[n] != 0)
	{
	  msb = tc_part_msb (parts[n]);

	  return msb + n * t_integer_part_width;
	}
    }
  while (n);

  return 0;
}

/* DST = LHS + RHS + C where C is zero or one.  Returns the carry
   flag.  */
t_integer_part
APInt::tc_add (t_integer_part *dst, const t_integer_part *lhs,
	      const t_integer_part *rhs, t_integer_part c, unsigned int parts)
{
  unsigned int i;

  assert (c <= 1);

  for (i = 0; i < parts; i++)
    {
      t_integer_part l;

      l = lhs[i];
      if (c)
	{
	  dst[i] = l + rhs[i] + 1;
	  c = (dst[i] <= l);
	}
      else
	{
	  dst[i] = l + rhs[i];
	  c = (dst[i] < l);
	}
    }

  return c;
}

/* DST = LHS - RHS - C where C is zero or one.  Returns the carry
   flag.  */
t_integer_part
APInt::tc_subtract (t_integer_part *dst, const t_integer_part *lhs,
		   const t_integer_part *rhs, t_integer_part c,
		   unsigned int parts)
{
  unsigned int i;

  assert (c <= 1);

  for (i = 0; i < parts; i++)
    {
      t_integer_part l;

      l = lhs[i];
      if (c)
	{
	  dst[i] = l - rhs[i] - 1;
	  c = (dst[i] >= l);
	}
      else
	{
	  dst[i] = l - rhs[i];
	  c = (dst[i] > l);
	}
    }

  return c;
}

/*  DST += SRC * MULTIPLIER + PART   if add is true
    DST  = SRC * MULTIPLIER + PART   if add is false

    Requires 0 <= DST_PARTS <= SRC_PARTS + 1.  If DST overlaps SRC
    they must start at the same point, i.e. DST == SRC.

    If DST_PARTS == SRC_PARTS + 1 no overflow occurs and zero is
    returned.  Otherwise DST is filled with the least significant
    DST_PARTS parts of the result, and if all of the omitted higher
    parts were zero return zero, otherwise overflow occurred and
    return one.  */
int
APInt::tc_multiply_part (t_integer_part *dst, const t_integer_part *src,
			t_integer_part multiplier, t_integer_part carry,
			unsigned int src_parts, unsigned int dst_parts,
			bool add)
{
  unsigned int i, n;

  /* Otherwise our writes of DST kill our later reads of SRC.  */
  assert (dst <= src || dst >= src + src_parts);
  assert (dst_parts <= src_parts + 1);

  /* N loops; minimum of dst_parts and src_parts.  */
  n = dst_parts < src_parts ? dst_parts: src_parts;

  for (i = 0; i < n; i++)
    {
      t_integer_part low, mid, high, src_part;

      /* [ LOW, HIGH ] = MULTIPLIER * SRC[i] + DST[i] + CARRY.

	 This cannot overflow, because

	    (n - 1) * (n - 1) + 2 (n - 1) = (n - 1) * (n + 1)

	 which is less than n^2.  */

      src_part = src[i];

      if (multiplier == 0 || src_part == 0)
	{
	  low = carry;
	  high = 0;
	}
      else
	{ 
	  low = low_half (src_part) * low_half (multiplier);
	  high = high_half (src_part) * high_half (multiplier);

	  mid = low_half (src_part) * high_half (multiplier);
	  high += high_half (mid);
	  mid <<= t_integer_part_width / 2;
	  if (low + mid < low)
	    high++;
	  low += mid;

	  mid = high_half (src_part) * low_half (multiplier);
	  high += high_half (mid);
	  mid <<= t_integer_part_width / 2;
	  if (low + mid < low)
	    high++;
	  low += mid;

	  /* Now add carry.  */
	  if (low + carry < low)
	    high++;
	  low += carry;
	}

      if (add)
	{
	  /* And now DST[i], and store the new low part there.  */
	  if (low + dst[i] < low)
	    high++;
	  dst[i] += low;
	}
      else
	dst[i] = low;

      carry = high;
    }

  if (i < dst_parts)
    {
      /* Full multiplication, there is no overflow.  */
      assert (i + 1 == dst_parts);
      dst[i] = carry;
      return 0;
    }
  else
    {
      /* We overflowed if there is carry.  */
      if (carry)
	return 1;

      /* We would overflow if any significant unwritten parts would be
	 non-zero.  This is true if any remaining src parts are
	 non-zero and the multiplier is non-zero.   */
      if (multiplier)
	for (; i < src_parts; i++)
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
APInt::tc_multiply (t_integer_part *dst, const t_integer_part *lhs,
		   const t_integer_part *rhs, unsigned int parts)
{
  unsigned int i;
  int overflow;

  assert (dst != lhs && dst != rhs);

  overflow = 0;
  tc_set (dst, 0, parts);

  for (i = 0; i < parts; i++)
    overflow |= tc_multiply_part (&dst[i], lhs, rhs[i], 0, parts,
				  parts - i, true);

  return overflow;
}

/* DST = LHS * RHS, where DST has twice the width as the operands.  No
   overflow occurs.  DST must be disjoint from both operands.  */
void
APInt::tc_full_multiply (t_integer_part *dst, const t_integer_part *lhs,
			const t_integer_part *rhs, unsigned int parts)
{
  unsigned int i;
  int overflow;

  assert (dst != lhs && dst != rhs);

  overflow = 0;
  tc_set (dst, 0, parts);

  for (i = 0; i < parts; i++)
    overflow |= tc_multiply_part (&dst[i], lhs, rhs[i], 0, parts,
				  parts + 1, true);

  assert (!overflow);
}

/* If RHS is zero QUOTIENT and REMAINDER are left unchanged, return
   one.  Otherwise set QUOTIENT to LHS / RHS with the fractional part
   discarded, set REMAINDER to the remainder, return zero.  i.e.

     LHS = RHS * QUOTIENT + REMAINDER

   SCRATCH is a bignum of the same size as the operands and result for
   use by the routine; its contents need not be initialized and are
   destroyed.  QUOTIENT, REMAINDER and SCRATCH must be distinct, and
   additionally SCRATCH cannot equal LHS.
*/
int
APInt::tc_divide (t_integer_part *quotient, t_integer_part *remainder,
		 const t_integer_part *lhs, const t_integer_part *rhs,
		 t_integer_part *srhs, unsigned int parts)
{
  unsigned int n, shift_count;
  t_integer_part mask;

  assert (quotient != remainder && quotient != srhs
		&& remainder != srhs && srhs != lhs);

  shift_count = tc_msb (rhs, parts);
  if (shift_count == 0)
    return true;

  shift_count = parts * t_integer_part_width - shift_count;
  n = shift_count / t_integer_part_width;
  mask = (t_integer_part) 1 << (shift_count % t_integer_part_width);

  tc_left_shift (srhs, rhs, parts, shift_count);
  tc_assign (remainder, lhs, parts);
  tc_set (quotient, 0, parts);

  /* Loop, subtracting SRHS if REMAINDER is greater and adding that to
     the total.  */
  for (;;)
    {
      int compare;

      compare = tc_compare (remainder, srhs, parts);
      if (compare >= 0)
	{
	  tc_subtract (remainder, remainder, srhs, 0, parts);
	  quotient[n] |= mask;
	}

      if (shift_count == 0)
	break;
      shift_count--;
      tc_right_shift (srhs, srhs, parts, 1);
      if ((mask >>= 1) == 0)
	mask = (t_integer_part) 1 << (t_integer_part_width - 1), n--;
    }

  return false;
}

/* Shift a bignum left COUNT bits.  Shifted in bits are zero.  There
   are no restrictions on COUNT.  */
void
APInt::tc_left_shift (t_integer_part *dst, const t_integer_part *src,
		     unsigned int parts, unsigned int count)
{
  unsigned int jump, shift;

  /* Jump is the inter-part jump; shift is is intra-part shift.  */
  jump = count / t_integer_part_width;
  shift = count % t_integer_part_width;

  while (parts > jump)
    {
      t_integer_part part;

      parts--;

      /* dst[i] comes from the two parts src[i - jump] and, if we have
	 an intra-part shift, src[i - jump - 1].  */
      part = src[parts - jump];
      if (shift)
	{
	  part <<= shift;
	  if (parts >= jump + 1)
	    part |= src[parts - jump - 1] >> (t_integer_part_width - shift);
	}

      dst[parts] = part;
    }

  while (parts > 0)
    dst[--parts] = 0;
}

/* Shift a bignum right COUNT bits.  Shifted in bits are zero.  There
   are no restrictions on COUNT.  */
void
APInt::tc_right_shift (t_integer_part *dst, const t_integer_part *src,
		      unsigned int parts, unsigned int count)
{
  unsigned int i, jump, shift;

  /* Jump is the inter-part jump; shift is is intra-part shift.  */
  jump = count / t_integer_part_width;
  shift = count % t_integer_part_width;

  /* Perform the shift.  This leaves the most significant COUNT bits
     of the result at zero.  */
  for (i = 0; i < parts; i++)
    {
      t_integer_part part;

      if (i + jump >= parts)
	part = 0;
      else
	{
	  part = src[i + jump];
	  if (shift)
	    {
	      part >>= shift;
	      if (i + jump + 1 < parts)
		part |= src[i + jump + 1] << (t_integer_part_width - shift);
	    }
	}

      dst[i] = part;
    }
}

/* Bitwise and of two bignums.  */
void
APInt::tc_and (t_integer_part *dst, const t_integer_part *lhs,
	      const t_integer_part *rhs, unsigned int parts)
{
  unsigned int i;

  for (i = 0; i < parts; i++)
    dst[i] = lhs[i] & rhs[i];
}

/* Bitwise inclusive or of two bignums.  */
void
APInt::tc_or (t_integer_part *dst, const t_integer_part *lhs,
	     const t_integer_part *rhs, unsigned int parts)
{
  unsigned int i;

  for (i = 0; i < parts; i++)
    dst[i] = lhs[i] | rhs[i];
}

/* Bitwise exclusive or of two bignums.  */
void
APInt::tc_xor (t_integer_part *dst, const t_integer_part *lhs,
	      const t_integer_part *rhs, unsigned int parts)
{
  unsigned int i;

  for (i = 0; i < parts; i++)
    dst[i] = lhs[i] ^ rhs[i];
}

/* Complement of a bignum.  */
void
APInt::tc_complement (t_integer_part *dst, const t_integer_part *rhs,
		     unsigned int parts)
{
  unsigned int i;

  for (i = 0; i < parts; i++)
    dst[i] = ~rhs[i];
}

/* Comparison (unsigned) of two bignums.  */
int
APInt::tc_compare (const t_integer_part *lhs, const t_integer_part *rhs,
		  unsigned int parts)
{
  while (parts)
    {
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

// Operations for soft-fp.

/* Increment a bignum, return the carry flag.  */
t_integer_part
APInt::tc_increment (t_integer_part *dst, unsigned int parts)
{
  unsigned int i;

  for (i = 0; i < parts; i++)
    if (++dst[i] != 0)
      break;

  return i == parts;
}

void
APInt::tc_set_lsbs (t_integer_part *dst, unsigned int parts, unsigned int bits)
{
  unsigned int i;

  i = 0;
  while (bits > t_integer_part_width)
    {
      dst[i++] = ~(t_integer_part) 0;
      bits -= t_integer_part_width;
    }

  if (bits)
    dst[i++] = ~(t_integer_part) 0 >> (t_integer_part_width - bits);

  while (i < parts)
    dst[i++] = 0;
}
