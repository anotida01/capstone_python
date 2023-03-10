#include <stdio.h>
#include <stdlib.h>

#define N 1
#define C 48
#define H 1920
#define W 1080
#define B 4

// first reshape;
int d0 = 1;
int d1 = 3;
int d2 = 4;
int d3 = 4;
int d4 = 244;
int d5 = 244;

float first_reshape_flat[2857728] = {0};
float first_reshape[1][3][4][4][244][244];
float transpose_arr[1][3][244][4][244][4];
float transpose_out[2857728];
float final_arr[1][3][976][976];

void d2s(float input[1][48][244][244], float output[1][3][976][976])
{
    int n = 1;
    int c = 48;
    int h = 244;
    int w = 244;
    int b = 4;
    int out_c = c / (b * b);
    int out_h = h * b;
    int out_w = w * b;
    


    //flatten


    int reshape_count = 0;
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 48; j++)
        {
            for (int k = 0; k < 244; k++)
            {
                for (int z = 0; z < 244; z++)
                {
                    first_reshape_flat[reshape_count] += input[i][j][k][z];
                    reshape_count++;
                }
            }
        }
    }
    reshape_count = 0;
    //rebuild
    /*
    for (int i0 = 0; i < d0; i++)
    {
        for (int i1 = 0; i < d1; i++)
        {
            for (int i2 = 0; i < d2; i++)
            {
                for (int i3 = 0; i < d3; i++)
                {
                    for (int i4 = 0; i < d4; i++)
                    {
                        for (int i5 = 0; i < d5; i++)
                        {
                            first_reshape[i0][i1][i2][i3][i4][i5] = first_reshape_flat[reshape_count];
                            reshape_count ++;
                        }
                    }
                }
            }
        }
    }
    */
    //transpose
    for (int i = 0; i < d0; i++)
    {
        for (int j = 0; j < d1; j++)
        {
            for (int k = 0; k < d3; k++)
            {
                for (int l = 0; l < d2; l++)
                {
                    for (int m = 0; m < d4; m++)
                    {
                        for (int n = 0; n < d5; n++)
                        {
                            transpose_out[(((((i * d1 + j) * d3 + k) * d2 + l) * d4 + m) * d5 + n) * d1] = first_reshape_flat[(((((k * d4 + m) * d1 + j) * d2 + l) * d3 + i) * d5 + n) * d0];
                        }
                    }
                }
                printf("in transpose: %f", transpose_out[j]);
            }
        }
    }
    //rebuild
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 976; k++)
            {
                for (int z = 0; z < 976; z++)
                {
                    output[i][j][k][z] = transpose_out[reshape_count];
                }
            }
        }
                        printf("in second reshape: %f", output[i][0][0][0]);

    }
}