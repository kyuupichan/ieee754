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
  APFloat d (APFloat::IEEEsingle, APFloat::fcZero, false);

  for (int i = 0; i < 100000; i++)
    d.convertFromString("1.2347896543216459876543216549879876464613121321654654654987987946546453213213216546546549789879876546546129816513523478965432164598765432165498798764646131213216546546549879879465464532132132165465465497898798765465461298165135123478965432164598765432165498798764646131213216546546549879879465464532132132165465465497898798765465461298165135234789654321645987654321654987987646461312132165465465498798794654645321321321654654654978987987654654612981651351234789654321645987654321654987987646461312132165465465498798794654645321321321654654654978987987654654612981651352347896543216459876543216549879876464613121321654654654987987946546453213213216546546549789879876546546129816513512347896543216459876543216549879876464613121321654654654987987946546453213213216546546549789879876546546129816513523478965432164598765432165498798764646131213216546546549879879465464532132132165465465497898798765465461298165135123478965432164598765432165498798764646131213216546546549879879465464532132132165465465497898798765465461298165135234789654321645987654321654987987646461312132165465465498798794654645321321321654654654978987987654654612981651351e309", APFloat::rmNearestTiesToEven);

  char s[400];
  d.convertToHexString(s, 0, false, APFloat::rmNearestTiesToEven);

  printf("%s\n", s);
  return 0;
}