//
//  bitarray.h
//  C-2015
//
//  Created by Michael Hopkins on 26/02/2020.
//  Copyright (c) 2020 Michael Hopkins. All rights reserved.
//

#ifndef C_2015_bitarray_h
#define C_2015_bitarray_h

/*
	 https://stackoverflow.com/questions/2525310/how-to-define-and-work-with-an-array-of-bits-in-c
	 https://stackoverflow.com/questions/44978126/structure-for-an-array-of-bits-in-c
	 http://c-faq.com/misc/bitsets.html
*/

#include <limits.h>		/* for CHAR_BIT */
#define BIT_ARRAY_BITSIZE 32     // 
#define BIT_ARRAY_SHIFT 5        // 5 for 32, 6 for 64
#define BIT_ARRAY_TYPE uint32_t
// use uint8_t to guarantee 8, uint32_t to minimise number of cells required and possibly speed up calculations - yes it does?

#define BITMASK(	b	) 		( 1 << ((b) % BIT_ARRAY_BITSIZE ))
#define BITSLOT(	b	) 		((b) >> BIT_ARRAY_SHIFT ) // ((b) / BIT_ARRAY_BITSIZE ) //

#define BITSET(	a, b	) 	((a)[BITSLOT(b)] |=  BITMASK(b))
#define BITCLEAR(	a, b	) 	((a)[BITSLOT(b)] &= ~BITMASK(b))
#define BITTEST(	a, b	) 	((a)[BITSLOT(b)] &   BITMASK(b))

#define BITNSLOTS(	nb	) 	((nb + BIT_ARRAY_BITSIZE - 1) / BIT_ARRAY_BITSIZE)


/*
 Here are some usage examples. To declare an ``array'' of 47 bits:
 
 	uint8_t bitarray[BITNSLOTS(47)];
 
 To set the 23rd bit:
 	BITSET(bitarray, 23);
 
 To test the 35th bit:
 	if(BITTEST(bitarray, 35)) ...
 
 To compute the union of two bit arrays and place it in a third array (with all three arrays declared as above):
 	for(i = 0; i < BITNSLOTS(47); i++)
 		array3[i] = array1[i] | array2[i];
 
 To compute the intersection, use & instead of |.
 
 As a more realistic example, here is a quick implementation of the Sieve of Eratosthenes, for computing prime numbers:
 
 	#include <stdio.h>
	#include <string.h>
 
 	#define MAX 10000
 
 	int main() {
 		
 		char bitarray[BITNSLOTS(MAX)];
 		int i, j;
 
 		memset(bitarray, 0, BITNSLOTS(MAX));
 
 		for(i = 2; i < MAX; i++) {
 			
 			if(!BITTEST(bitarray, i)) {
 
 				printf("%d\n", i);
 
 				for(j = i + i; j < MAX; j += i)
 					BITSET(bitarray, j);
				}
 			}
 	
 	return 0;
 	}

 */

/*
 
 So, to set the kth bit in array A:
 
 void  SetBit( int A[],  int k )
 {
	 int i = k/32;        //gives the corresponding index in the array A
	 int pos = k%32;      //gives the corresponding bit position in A[i]
	 
	 unsigned int flag = 1;   // flag = 0000.....00001
	 
	 flag = flag << pos;      // flag = 0000...010...000   (shifted k positions)
	 
	 A[i] = A[i] | flag;      // Set the bit at the k-th position in A[i]
 }
 or in the shortened version
 
 void  SetBit( int A[],  int k )
 {
 	A[k/32] |= 1 << (k%32);  // Set the bit at the k-th position in A[i]
 }
 similarly to clear kth bit:
 
 void  ClearBit( int A[],  int k )
 {
 	A[k/32] &= ~(1 << (k%32));
 }
 and to test if the kth bit:
 
 int TestBit( int A[],  int k )
 {
 	return ( (A[k/32] & (1 << (k%32) )) != 0 ) ;
 }
 As said above, these manipulations can be written as macros too:
 
 #define SetBit(A,k)     ( A[(k/32)] |= (1 << (k%32)) )
 #define ClearBit(A,k)   ( A[(k/32)] &= ~(1 << (k%32)) )
 #define TestBit(A,k)    ( A[(k/32)] & (1 << (k%32)) )
 */
#endif
