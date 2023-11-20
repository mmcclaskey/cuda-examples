# Breaking MD5 hashes on the GPU using CUDA by NVIDIA
High end video cards are have hundreds of processing cores cable of managing thousands of threads. Until recent years it has been difficult for developers and scientific researchers to utilize the power of the GPU for highly parallel tasks. With the release of the CUDA software development kit (SDK) NVidia has allowed programmers to take advantage of relatively cheap GPUs for executing highly parallel algorithms. CUDA breaks code execution into device code, code running on the video card, and host code, code running on the CPU. The device code is written in C and compiled by NVidia's compiler while the host code can be written any number of languages using NVidia's API (NVIDIA, 2008).

JCuda is a library that provides Java bindings for the Cuda API. This allows Java programmers to harness the power of the GPU to execute code. In this case Java will be used to manage jobs pushed to the graphics card. The GPU algorithm has been designed to hash 95 to the power of 5 passwords on each run, this equates to the first 5 characters of a password. Each thread has two nested loops which iterate through the last two characters of the 5 character password. The first three characters are extrapolated from the thread and block IDs. Each thread on each block has access to built-in variables that tell the thread what its thread index and block indexes are. Blocks can be shaped in 1, 2, or 3 dimensions, therefore, the threadID is a vector which is inherent in the type of processing that a GPU typically does. The variable that contains the treadID is threadIdx.x, threadIdx.y, and threadIdx.z, in a one dimensional block of 128 threads you would only use threadIdx.x. Blocks can be arranged in 1 or 2 dimensions; the block index of the current thread is accessed through the variable blockIdx.x and blockIdx.y. To illustrate the thread management and use of the built-in thread and block index variables the following code simultaneously tests hundreds of numbers to see if they are a prime number.

```C
extern "C" __global__ void
Parallel_isPrime(const unsigned long long int * input, unsigned long long int * output)
{
      int tid = threadIdx.x;
      int dib = blockIdx.x;
   
      int index = (dib * input[1]) + tid;
      unsigned long long int num = input[0];

      //new method
      if ((tid % 2) == 0)
          num += (((dib * input[1]) + tid) / 2) * 6;
     else
         num += ((((dib * input[1]) + (tid - 1)) / 2) * 6) + 2;
 
     //find the maximum number to loop through
     unsigned long long int max = rint(sqrt((double)num));
     output[index] = num;
  
     for (unsigned long long int i = 3; i <= max; i+=2)
     {
          if ((num % i) == 0)
          {   
              output[index] = 0;
              break;  
           }
      }
 }
```
				
The premise behind the function is simple, starting with the integer passed in input[0], have each thread test a different number to see if it is a prime number. The first thing the function must do is determine which number it will test based on what thread ID and block ID the instance is in, as seen in lines 4 and 5. Next an index value is needed that will be unique for each thread on each block, this index will be the index in the output array; each thread will write to the output at its unique index. Input[1] contains the number of threads per block, so the block ID * the number of threads per block plus the thread ID within the current block provides a unique number for each instance of this function. Next, each instance of the function needs a number to test starting with the number provided in input[0]. Test input[0] + index would be a simple method to have each instance of the function test a different number starting at the provided number, however, this method would waste a lot of threads on numbers known not to be prime. In base 10 all prime numbers are odd numbers with the exception of the number 2, skipping even numbers would cut the workload by 50%. In a similar fashion, using base 6 instead of base 10 reduces the workload by 67%. Lines 11 through 14 ensure that each instance tests a different number, skipping numbers that are known not to be prime. Finally, the loop starts at the number 3 and performs a modulus function attempting to find a number that will evenly divide into the number in question.

CUDA threads have access to multiple memory spaces; per thread local memory, per block shared memory, and global memory (NVIDIA, 2008). In addition to these three tiers there are also two read-only memory spaces accessible to all threads: constant and texture memory space. The simple prime function, the parameters input and output are accessible to all threads while all variables declared inside the function are unique to each instance of the function. Variables declared outside a function use either shared memory or global memory. Shared memory is faster and is shared with all threads in a block where global memory uses slower memory that is shared between all threads on all blocks.

A more useful application of this power would be for breaking a hash used to compare passwords. The security of a password relies on its length and complexity. Complexity is achieved by using special characters and numbers thus increasing the character set. If a system only allowed for lower case characters to be used in a password, the password would be base 26. An eight character password in base 26 has 26 to the power of 8 combinations (208.8 billion). An eight character password with a full 95 character set of upper, lower, numbers, and special characters has 95 to the power of 8 possible combinations, roughly 32,000 times more that base 26. Traditionally this has made it impossible to fully brute force passwords. A brute force password attack is an attempt to enumerate every possible combination, hashing each one, and comparing it to the hash of the actual password. To get around this computational issue hackers and security professionals use libraries of known common passwords. This form of attack relies on the user having a common password or a password comprised of words found in a dictionary. Substituting letters such as "a" with "@" can be accounted for in these attacks thus raising the bar. Even with a large library and complex substitution rules, the library attack cannot break every password. In order to break a password of reasonable complexity you need the capability to compute millions of hashes per second. A salt is often used to increase the security of hash algorithms. A salt is a string that is appended to the password before hashing. The salt can be randomly generated, static, or a timestamp, but in all cases the salt has to be stored somewhere. Cisco level 5 passwords are shown in the configuration as a hash with the salt shown in plain text.

The Parrallel_Hash algorithm in the code section below is based the original RSA algorithm and a C++ version written by Frank Thilo in C++ . The transform method of the md5 hash algorithm requires that the input be 512 bits long, therefore, prior to being hashed the original input must be padded if smaller than 512 bits. If the data is larger than 512 bits then the data must be processed in blocks and transformed several times. Since passwords will always be less than 512 bits the update and finalize functions are not required. After this and other enhancements to the algorithm were completed a 1,586% increase in speed was realized.

```C
/* MD5
Original algorithm by RSA Data Security, Inc
Adapted for NVIDIA CUDA by Matthew McClaskey
 
Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
rights reserved.
 
License to copy and use this software is granted provided that it
is identified as the "RSA Data Security, Inc. MD5 Message-Digest
Algorithm" in all material mentioning or referencing this software
or this function.
 
License is also granted to make and use derivative works provided
that such works are identified as "derived from the RSA Data
Security, Inc. MD5 Message-Digest Algorithm" in all material
mentioning or referencing the derived work.
 
RSA Data Security, Inc. makes no representations concerning either
the merchantability of this software or the suitability of this
software for any particular purpose. It is provided "as is"
without express or implied warranty of any kind.
 
These notices must be retained in any copies of any part of this
documentation and/or software.
*/
 
const unsigned int S11 = 7;
const unsigned int S12 = 12;
const unsigned int S13 = 17;
const unsigned int S14 = 22;
const unsigned int S21 = 5;
const unsigned int S22 = 9;
const unsigned int S23 = 14;
const unsigned int S24 = 20;
const unsigned int S31 = 4;
const unsigned int S32 = 11;
const unsigned int S33 = 16;
const unsigned int S34 = 23;
const unsigned int S41 = 6;
const unsigned int S42 = 10;
const unsigned int S43 = 15;
const unsigned int S44 = 21;
 
const int blocksize = 4; //<--MD5 Block size (in 32bit integers)
const int NUMTHREADS = 95; //<--NUMBER OF THREADS PER BLOCK
const unsigned int pwdbitlen = 40; //<--number of bits in plain text
 
/* F, G, H and I are basic MD5 functions */
__device__ inline unsigned int F(unsigned int x, unsigned int y, unsigned int z) { return (((x) & (y)) | ((~x) & (z))); }
__device__ inline unsigned int G(unsigned int x, unsigned int y, unsigned int z) { return (((x) & (z)) | ((y) & (~z))); }
__device__ inline unsigned int H(unsigned int x, unsigned int y, unsigned int z) { return ((x) ^ (y) ^ (z)); }
__device__ inline unsigned int I(unsigned int x, unsigned int y, unsigned int z) { return ((y) ^ ((x) | (~z))); }
 
/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
 
/* Rotation is separate from addition to prevent recomputation */
__device__ inline void FF(unsigned int &a, unsigned int b, unsigned int c, unsigned int d, unsigned int x, unsigned int s, unsigned int ac)
{
	a = ROTATE_LEFT(a + F(b, c, d) + x + ac, s) + b;
}
 
__device__ inline void GG(unsigned int &a, unsigned int b, unsigned int c, unsigned int d, unsigned int x, unsigned int s, unsigned int ac)
{
	a = ROTATE_LEFT(a + G(b, c, d) + x + ac, s) + b;
}
 
__device__ inline void HH(unsigned int &a, unsigned int b, unsigned int c, unsigned int d, unsigned int x, unsigned int s, unsigned int ac)
{
	a = ROTATE_LEFT(a + H(b ,c ,d) + x + ac, s) + b;
}
 
__device__ inline void II(unsigned int &a, unsigned int b, unsigned int c, unsigned int d, unsigned int x, unsigned int s, unsigned int ac)
{
	a = ROTATE_LEFT(a + I(b, c, d) + x + ac, s) + b;
}
 
__device__ void encode(unsigned char output[], const unsigned int input[], unsigned int len)
{
    for (unsigned int i = 0, j = 0; j < len; i++, j += 4)
    {
        output[j] = (unsigned char)(input[i] & 0xff);
        output[j + 1] = (unsigned char)((input[i] >> 8) & 0xff);
        output[j + 2] = (unsigned char)((input[i] >> 16) & 0xff);
        output[j + 3] = (unsigned char)((input[i] >> 24) & 0xff);
    }
}
 
extern "C" __global__ void Parrallel_Hash(unsigned int *input, char *output)
{  
	unsigned int a, b, c, d;

	//plain text int array
	unsigned int x[3];
	unsigned int total = 32;
	total += threadIdx.x;
   
	//Block Dim 9025 x 1
	total += ((blockIdx.x / 95) + 32) * 256;
	total += ((blockIdx.x % 95) + 32) * 65536;
   
	//Block Dim 95 x 95
	//total += (blockIdx.x + 32) * 256;
	//total += (blockIdx.y + 32) * 65536;
   
	//set ending integer of buffer
	//x[1] = input[4]; //up to 8 char password, last 4 char set by host input
	//x[1] = 128; //4 char password
	x[2] = 0;
               
	//loop sets 4th char
	for (unsigned int t = 32; t < 127; t++)
	{
		x[0] = total + (t * 16777216);
               
		//loop sets 5th char
		for (unsigned int u = 32; u < 127; u++)
		{
			x[1] = u + input[4]; //sets 5th through 8th char

			//load magic numbers
			a = 0x67452301;
			b = 0xefcdab89;
			c = 0x98badcfe;
			d = 0x10325476;

			// // Round 1
			FF ( a, b, c, d, x[ 0], S11, 3614090360); /* 1 */
			FF ( d, a, b, c, x[ 1], S12, 3905402710); /* 2 */
			FF ( c, d, a, b, x[ 2], S13,  606105819); /* 3 */
			FF ( b, c, d, a, 0, S14, 3250441966); /* 4 */
			FF ( a, b, c, d, 0, S11, 4118548399); /* 5 */
			FF ( d, a, b, c, 0, S12, 1200080426); /* 6 */
			FF ( c, d, a, b, 0, S13, 2821735955); /* 7 */
			FF ( b, c, d, a, 0, S14, 4249261313); /* 8 */
			FF ( a, b, c, d, 0, S11, 1770035416); /* 9 */
			FF ( d, a, b, c, 0, S12, 2336552879); /* 10 */
			FF ( c, d, a, b, 0, S13, 4294925233); /* 11 */
			FF ( b, c, d, a, 0, S14, 2304563134); /* 12 */
			FF ( a, b, c, d, 0, S11, 1804603682); /* 13 */
			FF ( d, a, b, c, 0, S12, 4254626195); /* 14 */
			FF ( c, d, a, b, pwdbitlen, S13, 2792965006); /* 15 */
			FF ( b, c, d, a, 0, S14, 1236535329); /* 16 */

			// Round 2
			GG (a, b, c, d, x[ 1], S21, 0xf61e2562); // 17
			GG (d, a, b, c, 0, S22, 0xc040b340); // 18
			GG (c, d, a, b, 0, S23, 0x265e5a51); // 19
			GG (b, c, d, a, x[ 0], S24, 0xe9b6c7aa); // 20
			GG (a, b, c, d, 0, S21, 0xd62f105d); // 21
			GG (d, a, b, c, 0, S22,  0x2441453); // 22
			GG (c, d, a, b, 0, S23, 0xd8a1e681); // 23
			GG (b, c, d, a, 0, S24, 0xe7d3fbc8); // 24
			GG (a, b, c, d, 0, S21, 0x21e1cde6); // 25
			GG (d, a, b, c, pwdbitlen, S22, 0xc33707d6); // 26
			GG (c, d, a, b, 0, S23, 0xf4d50d87); // 27
			GG (b, c, d, a, 0, S24, 0x455a14ed); // 28
			GG (a, b, c, d, 0, S21, 0xa9e3e905); // 29
			GG (d, a, b, c, x[ 2], S22, 0xfcefa3f8); // 30
			GG (c, d, a, b, 0, S23, 0x676f02d9); // 31
			GG (b, c, d, a, 0, S24, 0x8d2a4c8a); // 32

			// Round 3
			HH (a, b, c, d, 0, S31, 0xfffa3942); // 33
			HH (d, a, b, c, 0, S32, 0x8771f681); // 34
			HH (c, d, a, b, 0, S33, 0x6d9d6122); // 35
			HH (b, c, d, a, pwdbitlen, S34, 0xfde5380c); // 36
			HH (a, b, c, d, x[ 1], S31, 0xa4beea44); // 37
			HH (d, a, b, c, 0, S32, 0x4bdecfa9); // 38
			HH (c, d, a, b, 0, S33, 0xf6bb4b60); // 39
			HH (b, c, d, a, 0, S34, 0xbebfbc70); // 40
			HH (a, b, c, d, 0, S31, 0x289b7ec6); // 41
			HH (d, a, b, c, x[ 0], S32, 0xeaa127fa); // 42
			HH (c, d, a, b, 0, S33, 0xd4ef3085); // 43
			HH (b, c, d, a, 0, S34,  0x4881d05); // 44
			HH (a, b, c, d, 0, S31, 0xd9d4d039); // 45
			HH (d, a, b, c, 0, S32, 0xe6db99e5); // 46
			HH (c, d, a, b, 0, S33, 0x1fa27cf8); // 47
			HH (b, c, d, a, x[ 2], S34, 0xc4ac5665); // 48

			// Round 4
			II (a, b, c, d, x[ 0], S41, 0xf4292244); // 49
			II (d, a, b, c, 0, S42, 0x432aff97); // 50
			II (c, d, a, b, pwdbitlen, S43, 0xab9423a7); // 51
			II (b, c, d, a, 0, S44, 0xfc93a039); // 52
			II (a, b, c, d, 0, S41, 0x655b59c3); // 53
			II (d, a, b, c, 0, S42, 0x8f0ccc92); // 54
			II (c, d, a, b, 0, S43, 0xffeff47d); // 55
			II (b, c, d, a, x[ 1], S44, 0x85845dd1); // 56
			II (a, b, c, d, 0, S41, 0x6fa87e4f); // 57
			II (d, a, b, c, 0, S42, 0xfe2ce6e0); // 58
			II (c, d, a, b, 0, S43, 0xa3014314); // 59
			II (b, c, d, a, 0, S44, 0x4e0811a1); // 60
			II (a, b, c, d, 0, S41, 0xf7537e82); // 61
			II (d, a, b, c, 0, S42, 0xbd3af235); // 62
			II (c, d, a, b, x[ 2], S43, 0x2ad7d2bb); // 63
			II (b, c, d, a, 0, S44, 0xeb86d391); // 64

			a += 0x67452301;
			b += 0xefcdab89;
			c += 0x98badcfe;
			d += 0x10325476;

			//check if hash matches
			if (a == input[0] &&
					b == input[1] &&
					c == input[2] &&
					d == input[3])
					{
							unsigned char firstInt[8];
							encode(&firstInt[0], &x[0], 8);

							*output = firstInt[0];
							*(output + 1) = firstInt[1];
							*(output + 2) = firstInt[2];
							*(output + 3) = firstInt[3];
							*(output + 4) = firstInt[4];
							*(output + 5) = firstInt[5];
							*(output + 6) = firstInt[6];
							*(output + 7) = firstInt[7];
					}
		}
	}
}
```
				
The code section at the end of the article shows the host code in Java. Lines 31 through 40 first get a string hash in the standard hexadecimal format from the user, that hash is then converted into 4 integers which is stored in the first 4 elements of the hashin array. The Parallel_Hash algorithm enumerates every possible combination of the first 5 characters, the 5th element of the hashin variable allows the Java programmer to pass in any additional characters, up to 8, in the form of an integer. On line 46 hashin[4] is set to 128 which is end of string. In other words, no additional characters where passed in. Further work with the Java code is needed to execute the Parallel_Hash algorithm multiple times on multiple GPUs while incrementing hashin[4] to enumerate every combination of passwords larger than 5 characters. The Parallel_Hash function is setup to break 5 to 8 character passwords and could easily handle any size up to 512 bits, however, for testing only one batch is performed which equates to every possible enumeration of a 5 character password. Line 49 calls a method that will compile the CUDA C code with the NVidia nvcc compiler. Next a device object is created using index 0, if the computer had multiple CUDA capable video cards the 0 could be changed to select which card to use. The proceeding code is apparent in the remarks of the code, the kernel is loaded to the device, a pointer to the function is created, memory is allocated for the input and output, pointers to each are created, input is transferred to the device, parameters are defined using the pointers, the function is executed, and finally output is received and displayed to the user. To attempt to break passwords larger than 5 characters, a loop around the CUDA function call would have to be created and the hashin[4] integer would have to enumerated inside the loop.

The CUDA device code on attachment 1 starts by declaring the variable to hold the plaintext password called x[3]. Three integers are enough to hold a 12 character password with the first integer holding the first 4 characters, of which the first three are extrapolated from the thread and block IDs. In line 100 the thread ID is used for the first character, since each block has 95 threads this equates to 0-94 added to totals current value of 32. This provides a value of 32 through 127, the ASCII codes for the character set in test. Next we extrapolate the second and third character from the block ID. Since 9025 blocks are being executed (95 to the power of two), we have enough blocks to uniquely extract two characters from the block ID. The 2nd character is extrapolated using block ID divided by 95 plus 32 and the third is extrapolated using the block ID mod 95 plus 32. They are then multiplied by 256 and 65536 respectively to essentially shift the bits to the left and added to total. Characters 4 and 5 are provided by two nested loops. From here each plaintext password is ran through the 4 rounds of 16 transforms that comprise the MD5 hash algorithm and compared to the hash provided by the Java host code.

The Parrallel_Hash algorithm running on a NVidia 8800GT with 112 CUDA cores can hash 325,802,500 passwords per second. The 8800GT is a mid level card introduced in 2007; by contrast a late model mid-tier GTX 570 has 480 CUDA cores at a clock speed of 1,464MHz each. Testing on other cards shows a nearly linear increase in performance per CUDA core with only mild increases in speed with higher memory bus and CPU speeds. It can be safely assumed that running this algorithm on a GTX 570 will see at minimum a 400% increase in speed; about 1,303,210,000 hashes per second. This level of performance is vastly greater than that which can be achieved on any CPU to date. At this rate it would take a GTX 570 roughly two months to enumerate and hash every possible 8 character password in a 95 character set. In order to achieve more practical speeds, multiple GPUs will be needed and host software will be required to manage those resources. JCUDA allows the developer to use Java to select a desired video card, pass parameters, execute the parallel hash function, and get the results. Using Java the execution and flow control of jobs can be managed between GPUs, on a single platform or even across the internet in a distributed fashion. Utilizing 4 relatively cheap NVIDIA GTX 570 video cards on a single system could increase performance to fully enumerate an 8 character password in just 15 days. The implication of this study is that the MD5 hash algorithm may no longer be suitable to protect user passwords. Using a more complex algorithm that does twice as many operations would only double the time required to break it. By contrast, using a longer password raises the time required exponentially. Therefore it is imperative that systems require users to have passwords longer than 8 characters and have complexity rules to force special, number, upper, and lower case characters.

```java
package cudahash;
import java.io.*;
import jcuda.*;
import jcuda.driver.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.Scanner;
 
/**
 *
 * @author Matt
 */
public class CudaHash {
    //thread and block variables
    static final int NUM_BLOCKS_X = 9025;
    static final int NUM_BLOCKS_Y = 1;
    static final int NUM_THREADS_PER_BLOCK = 95;
    static final int PASSWORD_LEN = 8;
   
 
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException
    {
        //get input from user
        Scanner userin = new Scanner(System.in);
        String userinput = userin.nextLine();
        int[] tmp = hexToHash(userinput);
        int[] hashin = new int[5];
       
        //first 4 are the hash
        hashin[0] = tmp[0];
        hashin[1] = tmp[1];
        hashin[2] = tmp[2];
        hashin[3] = tmp[3];
        //the 5th number is difficult to explain
        //the GPU code enumerates every possible password in a 5 character password
        //this integer allows you to append more on to the end of each attempted password
        //the algorithm works on integers, not text so it must be converted
        //see documentation for further explanation
        hashin[4] = 128;
       
        //compile GPU code if required
        String cubinFileName = prepareCubinFile("hashgpuv3.cu", true);
 
        // Initialize the driver and create a context for the first device.
        JCudaDriver.cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        JCudaDriver.cuDeviceGet(dev, 0);
        JCudaDriver.cuCtxCreate(pctx, 0, dev);
       
        // Load the CUBIN file.
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, cubinFileName);
 
        // Obtain a function pointer to the "sampleKernel" function.
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "Parrallel_Hash");
       
        //allocate memory on device
        CUdeviceptr inPtr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(inPtr, hashin.length * Sizeof.INT);
       
        //transfer hash to device
        JCudaDriver.cuMemcpyHtoD(inPtr, Pointer.to(hashin), Sizeof.INT * 5);
       
        //allocate device output
        CUdeviceptr outPtr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(outPtr, PASSWORD_LEN * Sizeof.BYTE);
       
        //setup execution form (threads and blocks)
        JCudaDriver.cuFuncSetBlockShape(function, NUM_THREADS_PER_BLOCK, 1, 1);
       
        //set parameters
        Pointer dIn = Pointer.to(inPtr);
        Pointer dOut = Pointer.to(outPtr);
       
        //int offset = 0;
        //offset = JCudaDriver.align(offset, Sizeof.POINTER);
        JCudaDriver.cuParamSetv(function, 0, dIn, Sizeof.POINTER);
        //offset += Sizeof.POINTER;
       
        //offset = JCudaDriver.align(offset, Sizeof.POINTER);
        JCudaDriver.cuParamSetv(function, Sizeof.POINTER, dOut, Sizeof.POINTER);
        //offset += Sizeof.POINTER;
       
        JCudaDriver.cuParamSetSize(function, Sizeof.POINTER * 2);
       
        //call function
        JCudaDriver.cuLaunchGrid(function, NUM_BLOCKS_X, NUM_BLOCKS_Y);
        JCudaDriver.cuCtxSynchronize();
       
        //get output
        byte[] hostOut = new byte[PASSWORD_LEN];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOut), outPtr, PASSWORD_LEN * Sizeof.BYTE);
       
        //display result
        System.out.print("\n\n");
        for (int i = 0; i < hostOut.length; i++)
            System.out.print((char)hostOut[i]);
    }
   
    /**
     * Converts a user-friendly hex based hash to 4 integers
     * @param hash
     * @return
     * @throws Exception
     */
    public static int[] hexToHash(String hash) throws IOException
    {
       
        if (hash.length() != 32)
            throw new IOException("Invalid hash input.");
       
        int[] result = new int[4];
       
        String tmp;
        for (int i = 0; i < 32; i += 8)
        {
            //get next 4 bytes in reverse order
            tmp = hash.substring(i + 6, i + 8) +
                    hash.substring(i + 4, i + 6) +
                    hash.substring(i + 2, i + 4) +
                    hash.substring(i + 0, i + 2);
           
            //convert to integer
            result[(i + 1) / 8] = (int)Long.parseLong(tmp, 16);
           
            System.out.print(result[(i + 1) / 8] + "\n");
        }
       
        return result;
    }
   
    /**
     * modified by Matt McClaskey, based on example provided in JCUDA documentation
     * www.jcuda.org
     * The extension of the given file name is replaced with cubin.
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * cubin file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the CUBIN file
     * @throws IOException If an I/O error occurs
     */
    private static String prepareCubinFile(String cuFileName, Boolean overwrite) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String cubinFileName = cuFileName.substring(0, endIndex+1)+"cubin";
        System.out.print(cubinFileName);
        File cubinFile = new File(cubinFileName);
        System.out.print(cubinFile.getAbsolutePath());
        if (!overwrite && cubinFile.exists())
        {
            return cubinFileName;
        }
 
        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -arch sm_11 -cubin "+
            cuFile.getPath()+" -o "+cubinFileName;
 
        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);
 
        String errorMessage = new String(toByteArray(process.getErrorStream()));
        String outputMessage = new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while waiting for nvcc output", e);
        }
 
        System.out.println("nvcc process exitValue "+exitValue);
        if (exitValue != 0)
        {
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException("Could not create .cubin file: "+errorMessage);
        }
        return cubinFileName;
    }
   
    /**
     * this method was taken from JCuda documentation
     * www.jcuda.org
     * Fully reads the given InputStream and returns it as a byte array.
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
 
}
```
