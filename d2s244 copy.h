#include <stdio.h>
#include <stdlib.h>

#define N 1
#define C 48
#define H 1920
#define W 1080
#define B 4

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

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < out_c; j++)
        {
            for (int k = 0; k < out_h; k++)
            {
                for (int l = 0; l < out_w; l++)
                {
                    int in_c = j + (k / b) * b * out_c + (l / b) * b * out_c * c;
                    int in_h = k % b;
                    int in_w = l % b;
                    output[i][j][k][l] = input[i][in_c][in_h][in_w];
                }
            }
        }
    }
}