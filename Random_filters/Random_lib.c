
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "bitarray_new.h" // for SBC memories

//#define USE_OPENMP // comment out if OpenMP not available
#ifdef USE_OPENMP
#include <omp.h>
#endif

// utility macros to help keep code simple
#define FOR_LOOP( i, n ) for( (i) = 0; (i) < (n); (i)++ )

// define problem sizes - files are for MNIST and these sizes at the moment
#define W 1568  
#define D 10  

#define N1	6 
#define N2	8 
#define N3	10 
#define N4	12 

#define TRAIN_SZ 60000
#define TEST_SZ  10000
#define INPUT_SZ 784

#define SBC_CT 6 

#define TRUE  1
#define FALSE 0

#define K_WTA 10  // number of largest values to choose
#define THRESH_DIV_SHIFT 5

// these definitions updated 7th April 2022 for zero offset types with no wasted memory - testing
// Make vector on the heap with defined type: sz = length
#define HEAP_VEC( name, TYPE, sz ) TYPE *name; name = (TYPE *)calloc( (size_t)(sz), sizeof( TYPE ) ); if(!name) { printf(" allocation failure in heap_vector() "); exit(1); } 
#define FREE_VEC( name ) free( (void*)name );  // free vector memory

// Make matrix on the heap with defined type: szr = number of rows, szc = number of columns
#define HEAP_MAT( name, TYPE, szr, szc ) TYPE **name; name = (TYPE **) calloc( (size_t)( szr ), sizeof( TYPE* ) ); if( !name ) { printf(" allocation failure 1 in heap_matrix() "); exit(1); } name[0] = (TYPE *) calloc( (size_t)( szr * szc ), sizeof( TYPE ) ); if ( !name[0] ) { printf(" allocation failure 2 in heap_matrix() "); exit(1); } for( uint32_t i = 1; i < (szr); i++ ) name[i] = name[i-1] + szc;
#define FREE_MAT( name ) free( (void*)name[0] ); free( (void*)name );  // free matrix memory

// 3-Tensor type: szr = number of rows, szc = number of columns, szd = depth
#define HEAP_TENS( name, TYPE, szr, szc, szd ) TYPE ***name; name = (TYPE ***) calloc( (size_t)( szr ), sizeof( TYPE** ) ); if ( !name ) { printf(" allocation failure 1 in heap_tensor() "); exit(1); } else printf(" *** at %p ", &name ); name[0] = (TYPE **) calloc( (size_t)( szr * szc ), sizeof( TYPE* ) ); if ( !name[0] ) { printf(" allocation failure 2 in heap_tensor() "); exit(1); } else printf(" ** at %p ", &(name[0]) ); name[0][0] = (TYPE *) calloc( (size_t)( szr * szc * szd ), sizeof( TYPE ) ); if( !name[0][0] ) { printf(" allocation failure 3 in heap_tensor() "); exit(1); } else printf(" * at %p with %lu bytes \n", &(name[0][0]), ( szr * szc * szd ) * sizeof( TYPE ) ); for( uint32_t j = 1; j < (szc); j++ ) name[0][j] = name[0][j-1] + szd; for( uint32_t i = 1; i < (szr); i++ ) { name[i] = name[i-1] + szc; name[i][0] = name[i-1][0] + szc * szd; for( j = 1; j < (szc); j++ ) name[i][j] = name[i][j-1] + szd; }
#define FREE_TENS( name ) free( (void*) name[0][0] ); free( (void*) name[0] ); free( (void*) name );

// Make bit array on the heap with defined type
#define MAKE_BITARRAY( name, sz ) BIT_ARRAY_TYPE * name; name = calloc( BITNSLOTS( sz ), sizeof( BIT_ARRAY_TYPE ) ); if( name == NULL ) { printf(" %s could not be allocated \n", #name ); exit(1); }

uint64_t ws=W;
uint64_t ds=D;
uint64_t wd=W*D;
uint64_t ww=W*W;
uint64_t wwd=W*D; // used for memory calculations, need to be globals
// uint32_t i, j, k;
// uint64_t wd, w;

#define MAT_INDEX_2D( i, j, s1 )          ((i)*(s1) + (j))              // for accessing 2D bit matrix 
#define TENS_INDEX_3D( i, j, l, s1, s2 )  ((i) + (j)*(s1) + (l)*(s2))   // for accessing 3D bit tensor  
// for cases here  ws = W  wd = W * D
// #define D2( i, j ) 			((i)*ws + (j))
#define D3( i, j, k ) 		((i) + (j)*(wd) + (k)*ws)


// bubble sort - 1 pass 
void bubble( int64_t value[], uint32_t pos[], uint32_t j ) {

   uint32_t i, n;
   int64_t  c;
   
   for ( i = j; i > 0; i-- ) {
      if( value[i-1] > value[i] ) return;
      c = value[i-1]; value[i-1] = value[i]; value[i] = c;
      n = pos[i-1];     pos[i-1] = pos[i];     pos[i] = n;
      }

}
 
 
// k Winner-Take-All routine - find & sort the top k values in a set value[] of length len
// assumes uint32_t value[len], uint32_t pos[k]; pos[] is original position in value[]
void kWTA( uint32_t k, uint32_t len, int64_t value[], uint32_t pos[] )
{
   uint32_t i;
    
   for ( i = 0; i < k; i++) { pos[i] = i; bubble( value, pos, i ); }    // sort the first k values

   for ( i = k; i < len; i++ ) {                                        // sort the remaining values

      if ( value[i] > value[k-1] ) {
      
         value[k-1] = value[i]; pos[k-1] = i; bubble( value, pos, k-1 );
         
         } 
      }
}

void	find_AD_firing_pattern_KWTA_FILTER_C( uint32_t loop, uint32_t w, uint8_t fsz, int16_t* fpos, int32_t* fcoef, int8_t** sensory_input, int64_t* adrowsum, uint8_t* ad_row_fired )
{
	int16_t  i, j, rel_pos;
	int64_t 	count;

	FOR_LOOP( i, w ) {  // loop over all filters in this AD
		
		count = 0;                    // reset count and firing patterns
		ad_row_fired[ i ] = FALSE;
		
		FOR_LOOP( j, fsz ) {  	      // loop over all pixels in this filter
		
		   rel_pos = fpos[ j ] + i;   // pixel position calculated relative to filter centre
		
		   if( rel_pos >= 0 && rel_pos < 784 )  // only accumulate convolution if within the image
		      count += sensory_input[ loop ][ rel_pos ] * fcoef[ j ];

			}  // j over filter length
      
      adrowsum[ i ] = count;
      
//printf(" %u  %u  %lld \n", fsz, i, adrowsum[ i ] ); //fflush( stdout );
		}  // i over W
	
	HEAP_VEC( pos, uint32_t, w )        // make storage for positions
	
	kWTA( K_WTA, w, adrowsum, pos );    // find largest filter outputs

// set the firing patterns from highest values	
	FOR_LOOP( i, K_WTA ) 
	   ad_row_fired[ pos[i] ] = TRUE; // printf(" ^^ %u  %u  %u \n", fsz, i, pos[ i ] ); }
	
   FREE_VEC( pos )                     // free storage for positions
}

/*
	================================================
	Return vector of ADEs that fire in ad_row_fired
	================================================
	
	Most basic case with no Hebbian learning or noise added to pixels. Assumes all ADEs have same width per AD.
	
	Inputs:
	-------
	loop - position in training or test set (i.e. 60k elements in MNIST training, 10k elements in MNIST test)
	
	w - length of this AD
	
	n - number of synapses in ADE (for this AD)
	
	sensory_input - data length-by-data width pre-loaded matrix of input values (i.e. 0.255 for E/MNIST)
	
	ad_row_thresh - w vector of firing thresholds per ADE
	
	address_decoder - w-by-n matrix of sample positions from input data vector (i.e. 784 possible elements for E/MNIST).
	Importantly; positive values are excitatory synapses, negative values are inhibitory synapses.
	
	Return:
	-------
	ad_row_fired - w vector relating ADEs that fired = 1 and didn't fire = 0
	
*/
void	find_AD_firing_pattern_C( uint32_t loop, uint32_t w, uint8_t n, uint8_t** sensory_input, int32_t* ad_row_thresh, int32_t** address_decoder, uint8_t* ad_row_fired )
{
	uint32_t	i, j, total = 0;
	int32_t 	count, position;
	int8_t 	yang;
	int16_t  pixel;

	FOR_LOOP( i, w ) {  // loop over all ADEs in AD
		
		count = 0;
		
		FOR_LOOP( j, n ) {  	// loop over all synapses in this ADE
		
		   if( address_decoder[i][j] > 0 ) 
		      { yang =  64; position =  address_decoder[i][j]; }
		   else  
		      { yang = -64; position = -address_decoder[i][j]; }
		   
		   pixel = sensory_input[ loop ][ position-1 ];

		   count += ( pixel - 127 ) * yang;
		   
			}  // j over N
      
		ad_row_fired[ i ] = ( count >= ad_row_thresh[ i ] ? TRUE : FALSE );
		
		}  // i over W
}


void write_to_sbc_C( uint8_t* first, uint8_t* second, BIT_ARRAY_TYPE* mem, uint32_t w, uint8_t label )
{
	uint32_t i, j, internal_count = 0;
	uint64_t tens_index;

	FOR_LOOP( i, w ) {
		if( first[i] ) {
			
			FOR_LOOP( j, w )
				if( second[j] ) {
					
					tens_index = D3( i, j, label );
						
					if( !BITTEST( mem, tens_index ) ) {

						internal_count++;  

						BITSET( mem, tens_index );
						}
					}
			}
		}
		
	// return internal_count;
}


// read bits from SBC for inference
void read_from_sbc_C( uint8_t* first, uint8_t* second, BIT_ARRAY_TYPE* mem, uint32_t w, uint32_t* store )
{
	uint64_t i, j, k;
	uint64_t tens_index;
	uint8_t 	bit_test;

	FOR_LOOP( i, w )
		if( first[i] )
			
			FOR_LOOP( j, w )
				if( second [j] ) {

					FOR_LOOP( k, ds ) {
					// 
						tens_index = D3( i, j, k);
						
						bit_test = BITTEST( mem, tens_index ); 

                  if( bit_test ) (store[k])++;
//						store[k] += ( bit_test != 0 );
						}
						
					}
}


// show firing pattern of the AD from a given input
void show_firing_pattern( uint8_t* ad_fire, uint16_t sz )
{
   uint16_t i;

   FOR_LOOP( i, sz )
      printf( "%1u ", ad_fire[i] );
   
   printf("\n");
}


// show thresholds in the AD
void show_thresholds( int32_t* ad_thresh, uint16_t sz )
{
   uint16_t i;

   printf("\n");
   
   FOR_LOOP( i, sz )
      printf( "%d ", ad_thresh[i] );
   
   printf("\n");
}


// show an AD synapse sampling pattern
void show_AD( int32_t** ad, uint16_t w, uint8_t n )
{
   uint16_t i, j;

   printf("\n");
   
   FOR_LOOP( i, w ) {
   
      FOR_LOOP( j, n )
         printf( "%5d ", ad[i][j] );
      
      printf("\n");
      }
   
   printf("\n");
}

