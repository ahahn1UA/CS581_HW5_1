/*
 Name: Andrew Hahn
 Email: ahahn1@crimson.ua.edu
 Course Section: CS 581
 Homework #: 5
 To Compile: nvcc HW5.cu -o HW5
 To Run: ./HW5 5000 5000
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>

#define DEAD 0
#define ALIVE 1

#define BLOCK_SIZE 16 // Adjust block size as needed

__global__ void evolve(int *current, int *next, int size) {
    // Define shared memory with halo cells
    __shared__ int shared_current[(BLOCK_SIZE + 2)*(BLOCK_SIZE + 2)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx + 1; // +1 for padding
    int y = blockIdx.y * blockDim.y + ty + 1;

    int idx = y * (size + 2) + x;

    // Compute indices in shared memory
    int s_width = BLOCK_SIZE + 2;
    int s_idx = (ty + 1) * s_width + (tx + 1);

    // Initialize shared memory to DEAD
    shared_current[s_idx] = DEAD;

    // Load cell data into shared memory
    if (x <= size && y <= size) {
        shared_current[s_idx] = current[idx];
    }

    // Load halo cells into shared memory
    // Left and right halos
    if (tx == 0) { // Left halo
        if (x > 1) {
            shared_current[s_idx - 1] = current[idx - 1];
        } else {
            shared_current[s_idx - 1] = DEAD;
        }
    }
    if (tx == BLOCK_SIZE - 1 || x == size) { // Right halo
        if (x < size) {
            shared_current[s_idx + 1] = current[idx + 1];
        } else {
            shared_current[s_idx + 1] = DEAD;
        }
    }

    // Top and bottom halos
    if (ty == 0) { // Top halo
        if (y > 1) {
            shared_current[s_idx - s_width] = current[idx - (size + 2)];
        } else {
            shared_current[s_idx - s_width] = DEAD;
        }
    }
    if (ty == BLOCK_SIZE - 1 || y == size) { // Bottom halo
        if (y < size) {
            shared_current[s_idx + s_width] = current[idx + (size + 2)];
        } else {
            shared_current[s_idx + s_width] = DEAD;
        }
    }

    // Corner halos
    if (tx == 0 && ty == 0) { // Top-left corner
        if (x > 1 && y > 1) {
            shared_current[s_idx - s_width - 1] = current[idx - (size + 2) - 1];
        } else {
            shared_current[s_idx - s_width - 1] = DEAD;
        }
    }
    if (tx == 0 && (ty == BLOCK_SIZE - 1 || y == size)) { // Bottom-left corner
        if (x > 1 && y < size) {
            shared_current[s_idx + s_width - 1] = current[idx + (size + 2) - 1];
        } else {
            shared_current[s_idx + s_width - 1] = DEAD;
        }
    }
    if ((tx == BLOCK_SIZE - 1 || x == size) && ty == 0) { // Top-right corner
        if (x < size && y > 1) {
            shared_current[s_idx - s_width + 1] = current[idx - (size + 2) + 1];
        } else {
            shared_current[s_idx - s_width + 1] = DEAD;
        }
    }
    if ((tx == BLOCK_SIZE - 1 || x == size) && (ty == BLOCK_SIZE - 1 || y == size)) { // Bottom-right corner
        if (x < size && y < size) {
            shared_current[s_idx + s_width + 1] = current[idx + (size + 2) + 1];
        } else {
            shared_current[s_idx + s_width + 1] = DEAD;
        }
    }

    // Synchronize to ensure all threads have loaded their data
    __syncthreads();

    if (x <= size && y <= size) {
        int aliveNeighbors = 0;

        // Calculate the number of alive neighbors using shared memory
        aliveNeighbors += shared_current[s_idx - s_width - 1];
        aliveNeighbors += shared_current[s_idx - s_width];
        aliveNeighbors += shared_current[s_idx - s_width + 1];
        aliveNeighbors += shared_current[s_idx - 1];
        aliveNeighbors += shared_current[s_idx + 1];
        aliveNeighbors += shared_current[s_idx + s_width - 1];
        aliveNeighbors += shared_current[s_idx + s_width];
        aliveNeighbors += shared_current[s_idx + s_width + 1];

        // Apply the Game of Life rules
        if (shared_current[s_idx] == ALIVE) {
            next[idx] = (aliveNeighbors == 2 || aliveNeighbors == 3) ? ALIVE : DEAD;
        } else {
            next[idx] = (aliveNeighbors == 3) ? ALIVE : DEAD;
        }
    }
}

void initializeBoard(int *board, int size) {
    // Seed the random number generator
    srand(52);

    int totalSize = (size + 2) * (size + 2);

    // Initialize all cells to DEAD
    for (int i = 0; i < totalSize; i++) {
        board[i] = DEAD;
    }

    // Randomly set cells to ALIVE or DEAD
    for (int y = 1; y <= size; y++) {
        int row = y * (size + 2);
        for (int x = 1; x <= size; x++) {
            int idx = row + x;
            board[idx] = (rand() % 2 == 0) ? DEAD : ALIVE;
        }
    }
}

void writeBoardToFile(int *board, int size, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file %s for writing\n", filename);
        return;
    }
    for (int y = 1; y <= size; y++) {
        int row = y * (size + 2);
        for (int x = 1; x <= size; x++) {
            fprintf(fp, board[row + x] == ALIVE ? "O " : ". ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char *argv[]) {
    // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Check if the correct number of arguments are given
    if (argc != 3) {
        printf("Usage: %s <size of board> <max generations>\n", argv[0]);
        return 1;
    }

    // Create variables
    int size = atoi(argv[1]);
    int maxGenerations = atoi(argv[2]);

    // Allocate host memory
    int *h_board1 = (int *)malloc((size + 2) * (size + 2) * sizeof(int));
    int *h_board2 = (int *)malloc((size + 2) * (size + 2) * sizeof(int));

    // Initialize boards
    initializeBoard(h_board1, size);
    initializeBoard(h_board2, size); // Initialize board2 to avoid uninitialized memory

    // Allocate device memory
    int *d_board1, *d_board2;
    cudaMalloc((void **)&d_board1, (size + 2) * (size + 2) * sizeof(int));
    cudaMalloc((void **)&d_board2, (size + 2) * (size + 2) * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_board1, h_board1, (size + 2) * (size + 2) * sizeof(int), cudaMemcpyHostToDevice);

    int generation = 0;

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Run the game
    while (generation < maxGenerations) {
        // Launch the kernel
        evolve<<<gridSize, blockSize>>>(d_board1, d_board2, size);

        // Swap the boards
        int *temp = d_board1;
        d_board1 = d_board2;
        d_board2 = temp;

        generation++;
    }

    // Copy final board back to host
    cudaMemcpy(h_board1, d_board1, (size + 2) * (size + 2) * sizeof(int), cudaMemcpyDeviceToHost);

    // Write final board to file
    writeBoardToFile(h_board1, size, "outputs/final_board.txt");

    // Free device memory
    cudaFree(d_board1);
    cudaFree(d_board2);

    // Free host memory
    free(h_board1);
    free(h_board2);

    // End timer and calculate total time
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/1e6);
    printf("Total time taken: %f seconds\n", elapsed);

    return 0;
}
