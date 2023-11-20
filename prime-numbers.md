# What is Cuda
I think NVidia's own description describes it well enough, "CUDA is NVIDIA's parallel computing architecture that enables dramatic increases in computing performance by harnessing the power of the GPU (graphics processing unit)." http://www.nvidia.com/object/what_is_cuda_new.html

Basically they have developed a platform that makes it easier for developers to use the massive parallel computing power of GPUs. Of course NVidia is not alone on this front, ATI, OpenCL, and even PS3s are possible solutions when highly parallel floating point operations are required. In fact the US Air Force has a 1,700 PS3 super computer http://dvice.com/archives/2010/05/us-air-force-ge.php. I have chosen Cuda for several reasons. I have a spare NVidia GeForce 8800 GTS lying around for starters and there is a .NET wrapper for the CUDA class library which means I can write the kernel in C as required to be compiled to run on video card but I can write the host application in C#. Under CUDA the video card is considered the device which the kernel runs on and the host is the application running on the CPU. For all the juicy details about how CUDA works check out NVidia's documentation here http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/NVIDIA_CUDA_ProgrammingGuide.pdf

Why prime numbers
I chose calculating prime numbers as a learning project for the SDK. I needed something that was simple and easily verifiable. A prime number algorithm is a classic programming question that could be asked on an interview for example and it seemed like a good way to learn CUDA.

## The Kernel
I ended up with 5 different versions of the kernel algorithm. Eventually it came down to two different algorithms types, depending on how large the starting number was.

The first version works by taking a number (n) and dividing 0 through n to x number of threads. So for simplicity, if the number was 100 and you had 10 threads each thread would get ten numbers and perform if (mod % n == 0) on each. The other version instead takes a starting number (n) and assigns each thread a number starting with n. So if you have 256 threads and the starting number was 100, thread one would test 100, thread 2 would test 101 and so on.

Here is the most basic, non-threaded prime number test using C#. It checks 1000 numbers for prime starting at 100.

```C
Int64 start = 100;
for (Int64 n = start; n < start + 1000; n += 1)
{
	for (Int64 i = 2; i < n; i += 1)
	{
		if ((n % i) == 0)
			break;
		if (i == n)
			isPrime(n);
	}
}
```
					
There are several issues with this algorithm. First of all there is no need to check even numbers for prime so I make sure that the starting number (n) is odd and iterate by 2 to ensure all numbers tested are odd. The second issue is there is no reason for the loop to go past the square root of n. Why, because 100 / 50 is 2, there is no reason to test 50 because 2 would have already been tried. Next issue is that the inner loop is iterating by 1, any odd number will not be evenly divisible by an even number so we need to start the loop at 3 and iterate by 2.

```C
Int64 start = 101;
Int64 max;
for (Int64 n = start; n < start + 2000; n += 2)
{
    Max = Math.Sqrt(n);
    for (Int64 i = 2; i <= Max; i += 2)
    {
        if ((n % i) == 0)
            break;
        if (i == Max)
            isPrime(n);
    }
}
```
					
Results:

-Skipping odd n numbers = 100% increase in numbers tested, notice we now go to 2000 in the same number of loops.
-Skipping even numbers on inner loop = 50% decrease in loop iterations
-Stopping i loop at sqrt of n = ((n/100) - (sqrt(n) / n))% decrease in top most loop iterations if the number is not a prime number.

Base6
Imagine base 10 numbers written on a spreadsheet starting at 0, with 10 columns. All prime numbers will occur in columns 2, 3, 4, 6, 8, and 10, of course these are odd number columns. We already took advantage of the fact that odd numbers appear in only 5 of the 10 columns by iterating the inner loop by 2, but in base 6, prime numbers only occur in 2 of the 6 columns thus giving you a 67% increase in speed over the original algorithm.

The Kernel Code
```C
extern "C" __global__ void
Parallel_isPrime(const unsigned long long int * input, unsigned long long int * output)
{
    int tid = threadIdx.x;  //the threadId of this instance
    int dib = blockIdx.x;  //the blockId of this instance
    //input[0] = number to start with
    //input[1] = number of threads
    
    int index = (dib * input[1]) + tid; 
    unsigned long long int num = input[0];

    //new method, base6, even number threads get column 2 and odd threads get column 6 numbers in base 6
    if ((tid % 2) == 0)
        num += (((dib * input[1]) + tid) / 2) * 6;
    else
        num += ((((dib * input[1]) + (tid - 1)) / 2) * 6) + 2;
    
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
					
Essentially what I have done is taken even threads and assigned it numbers in column 1 and odd threads and assigned it numbers in column 3, all other numbers are ignored. The block of code under the "new method" comment is what extrapolates from base 10 what is in column 1 and 3 and assigns them to a thread. This algorithm uses the same principals of the proceeding CPU version in C# only it divides the outer loop to each thread. So if you started the kernel off with 128 threads by 512 blocks you would have a total of 65536 threads, so if you started n off at 101 it would start at 101 and assign each of those threads a number that is in base 6 columns 1 or 3. At the end the host application gets an array of numbers back that tested true for prime.

The Host Code
```C#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
 
namespace PrimeNumbers
{
    class Program
    {
        //GPU constants                defaults
        private const int NUM_BLOCKS = 1024;
        private const int NUM_THREADS = 256;
        private const int NUM_PER_TEST = 262144;
 
        static void Main(string[] args)
        {
            try
            {
                // Init CUDA, select 1st device.
                CUDA cuda = new CUDA(0, true);
 
                // load module
                if (!File.Exists(Path.Combine(Environment.CurrentDirectory, "prime_kernel2.cubin")))
                    return;
 
                cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "prime_kernel2.cubin"));
                CUfunction func = cuda.GetModuleFunction("Parallel_isPrime");
 
                UInt64[] input = new UInt64[4];
                input[0] = Convert.ToUInt32(Console.ReadLine()); //this number will be the starting point
                input[1] = NUM_THREADS;
                input[2] = NUM_PER_TEST;
                input[3] = NUM_BLOCKS;
 
                //Make sure input number is an odd number
                if ((input[0] % 2) == 0)
                    throw new Exception("Error, you need to input an odd number.");
 
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Start();
 
                CUdeviceptr dinput = cuda.CopyHostToDevice<UInt64>(input);
                CUdeviceptr doutput = cuda.Allocate((uint)(sizeof(UInt64) * NUM_PER_TEST));
 
                cuda.SetParameter(func, 0, (uint)dinput.Pointer);
                cuda.SetParameter(func, IntPtr.Size, (uint)doutput.Pointer);
                cuda.SetParameterSize(func, (uint)(IntPtr.Size * 2));
 
                //launch GPU Prime num
                cuda.SetFunctionBlockShape(func, NUM_THREADS, 1, 1);
                cuda.SetFunctionSharedSize(func, (uint)(sizeof(UInt64) * NUM_THREADS));
                cuda.Launch(func, NUM_BLOCKS, 1);
 
                //CPU Prime num
                //PrimeNumLauncher pnl = new PrimeNumLauncher(NUM_THREADS_CPU, NUM_TESTS_CPU / NUM_THREADS_CPU, input[0] + (NUM_BLOCKS * NUM_THREADS));
 
                UInt64[] output = new UInt64[NUM_PER_TEST];
                cuda.CopyDeviceToHost<UInt64>(doutput, output);
 
                cuda.Free(dinput);
                cuda.Free(doutput);
 
                sw.Stop();
 
                //write each prime number
                foreach (ulong prime in output.Where(x => x != 0))
                    Console.WriteLine(prime);
 
 
                Console.WriteLine(string.Format("GPU: {1} numbers tested in {0} ms. ({2} nps)", sw.ElapsedMilliseconds.ToString(), NUM_PER_TEST * 2, ((NUM_PER_TEST * 2) / sw.ElapsedMilliseconds) * 1000));
            }
            catch (Exception ex) { Console.WriteLine(ex.Message); }
 
            Console.ReadKey();
        }
    }
}
```
					
# Final Thoughts
This was a good test to learn the basics of Cuda, however, it is not really practical. Although it proved to be significantly faster than anything I could write to run on my CPU, it is limited to 64bit integers which for practical prime number finding are useless. The largest prime number known is 12,978,189 digits long (http://primes.utm.edu/largest.html) which would require going outside the confines of built in data types and writing custom binary division and modulus functions. Certainly possible but this was a learning exercise for Cuda.