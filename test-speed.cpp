/*
   Copyright 2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include <cstdio>
#include "APFloat.h"

using namespace llvm;

int main (void)
{
  APFloat d (APFloat::IEEEsingle, "0x1.0p53");
  unsigned long long ull;
  APFloat::opStatus status;

  status = d.convertToInteger(&ull, 64, false, APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opOK);

  char s[100];
  d.convertToHexString(s, 0, false, APFloat::rmNearestTiesToEven);

  printf("%s %llu\n", s, ull);
  return 0;
}
