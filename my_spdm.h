#pragma once
#include <string.h>
#include <assert.h>
#include <vector>
#include <cstdlib>
#include <immintrin.h>

template <typename T>
struct BSRMatrix
{
    int shape[2];
    int blocksize[2];
    int nrowptr;
    int nnz;
    int* rowptr;
    int* nnzidx;
    int* colidx;
    T* data;
};

template <typename T>
struct BSCMatrix
{
    int shape[2];
    int blocksize[2];
    int ncolptr;
    int nnz;
    int* nnzidx;
    int* colptr;
    int* rowidxs;
    T* data;
};

template <typename T>
BSRMatrix<T>* create_bsr_matrix(
    const T* dense_matrix, const int shape[2], const int block_size[2]
){
    const int blksize = block_size[0] * block_size[1];
    BSRMatrix<T>* bsr_matrix = new BSRMatrix<T>;
    bsr_matrix->shape[0] = shape[0];
    bsr_matrix->shape[1] = shape[1];
    assert(shape[0] % blocksize[0] == 0);
    assert(shape[1] % blocksize[1] == 0);
    std::vector<int> rowptr;
    std::vector<int> colidxs;
    for (int b_row = 0; b_row < bsr_matrix->shape[0] / blocksize[0]; b_row++) {
        for (int b_col = 0; b_col < bsr_matrix->shape[1] / blocksize[1]; b_col++) {
            bool is_zero = true;
            const T* dense_start = dense_matrix + b_row * blocksize[0] * shape[1] + b_col * blocksize[1];
            for (int i = 0; i < bsr_matrix->blocksize[0]; i++) {
                for (int j = 0; j < bsr_matrix->blocksize[1]; j++) {
                    if (dense_start[i*shape[1]+j] != 0) {
                        is_zero = false;
                        goto done_check_zero;
                    }
                }
            }
done_check_zero:
            if (!is_zero) {
                colidxs.push_back(b_col);
            }
        }
    }
    rowptr.push_back(colidxs.size());

    bsr_matrix->nrowptr = rowptr.size();
    bsr_matrix->rowptr = new int[rowptr.size()];
    for (int i = 0; i < bsr_matrix->nrowptr; i++) {
        bsr_matrix->rowptr[i] = rowptr[i];
    }

    bsr_matrix->nnz = colidxs.size();
    bsr_matrix->colidxs = new int[colidxs.size()];
    for (int i = 0; i < bsr_matrix->nnz; i++) {
        bsr_matrix->colidxs[i] = colidxs[i];
    } 

    bsr_matrix->nnzidxs = new int[rowptr.size()-1];
    int nnzidx = 0;
    for (int i = 0; i < bsr_matrix->nrowptr-1; i++) {
        bsr_matrix->nnzidxs[i] = nnzidx;
        nnzidx += bsr_matrix->rowptr[i+1] - bsr_matrix->rowptr[i];
    }

    int nnz_idx = 0;
    bsr_matrix->data = (T*)aligned_alloc(64, bsr_matrix->nnz * blksize * sizeof(T));
    for (int b_row = 0; b_row < bsr_matrix->nrowptr-1; b_row++) {
        for (int b_col_idx = bsr_matrix->rowptr[b_row]; b_col_idx < bsr_matrix->rowptr[b_row+1]; b_col_idx++, nnz_idx++) {
            int b_col = bsr_matrix->colidxs[b_col_idx];
            T* blkstart = bsr_matrix->data + nnz_idx*blksize;
            const T* dense_start = dense_matrix + b_row * blocksize[0] * shape[1] + b_col * blocksize[1];
            for (int i = 0; i < bsr_matrix->blocksize[0]; i++) {
                for (int j = 0; j < bsr_matrix->blocksize[1]; j++) {
                    blkstart[i * bsr_matrix->blocksize[1] + j] = dense_start[i * shape[1] + j];
                }
            }
        }
    }
    return bsr_matrix;
}

template <typename T>
BSCMatrix<T>* create_bsc_matrix(
    const T* dense_matrix, const int shape[2], const int block_size[2]
){
    BSRMatrix<T>* bsr = create_bsr_matrix(dense_matrix, shape, blocksize);
    BSCMatrix<T>* bsc = new BSCMatrix<T>;
    const int bs = blocksize[0] * blocksize[1];
    bsc->shape[0] = bsr->shape[0];
    bsc->shape[1] = bsr->shape[1];
    bsc->blocksize[0] = bsr->blocksize[0];
    bsc->blocksize[1] = bsr->blocksize[1];
    bsc->nnz = bsr->nnz;
    bsc->ncolptr = bsr->shape[1] / bsr->blocksize[1] + 1;
    bsc->data = (T*) aligned_alloc(64, bsr->nnz * bs * sizeof(T));
    bsc->colptr = new int[bsc->ncolptr];
    bsc->rowidxs = new int[bsr->nnz];

    //transpose, to reorder
    int ptr = 0;
    for (int b_col=0; b_col < bsc->ncolptr-1; b_col++){
        bsc->colptr[b_col] = ptr;
        for (int b_row = 0, nnz_idx = 0; b_row < bsr->nrowptr-1; b_row++) {
            for (int b_col_idx = bsr->rowptr[b_row]; b_col_idx < bsr->rowptr[b_row+1]; b_col_idx++, nnz_idx++) {
                if (b_col == bsr->colidxs[b_col_idx]) {
                    memcpy(bsc->data + ptr * bs, bsr->data + nnz_idx * bs, sizeof(T) * bs);
                }
            }
        }
    }
    bsc->colptr[b_col] = ptr;
    bsc->nnzidxs = new int[bsc->ncolptr-1];
    int nnzidx = 0; // nums of block each col
    for (int i = 0; i < bsc->ncolptr-1; i++) {
        bsc->nnzidxs[i] = nnzidx; 
        nnzidx += bsc->colptr[i+1] - bsc->colptr[i];
    }
    destroy_bsr_matrix(bsr);
    return bsc;
}

template <typename T>
void destroy_bsr_matrix(BSRMatrix<T>* bsr_matrix) {
    free(bsr_matrix->data);
    delete[] bsr_matrix->rowptr;
    delete[] bsr_matrix->nnzidx;
    delete[] bsr_matrix->colidx;
    delete bsr_matrix;
}

template <typename T>
void destroy_bsc_matrix(BSCMatrix<T>* bsc_matrix) {
    free(bsc_matrix->data);
    delete[] bsc_matrix->colptr;
    delete[] bsc_matrix->nnzidx;
    delete[] bsc_matrix->rowidx;
    delete bsc_matrix;
}

//4 in 64, 4 * 1x16 blocks as a group
struct GroupInfo {
    int start_row;
    __mmask64 mask;
}__attribute__ ((__packed__));

void reorder_bsr_int8_4x16(BSCMatrix<int8_t>* bsc) {
    const int block_col = bsc->blocksize[1];
    const int block_row = bsc->blocksize[0];
    const int block_size = block_row * block_col;
    int8_t* new_data = (int8_t*) aligned_alloc(64, bsc->nnz * block_size * sizeof(int8_t));

    const int8_t* block_start = bsc->data;
    int new_data_ptr = 0;
    //reorder
    for (int nnz_idx=0; nnz_idx < bsc->nnz; ++nnz_idx) {
        for (int col=0; col < block_col; ++col) {
            for (int row=0; row < block_row; ++row) {
                new_data[new_data_ptr++] = block_start[row * block_col + col];
            }
        }
        block_start += block_size;
    }
    //orig rowidxs=1,2.., now rowidxs=1xblock_row,2xblock_row
    for (int i = 0; i < bsc->nnz; ++i){
        bsc->rowidxs[i] *= block_row;
    }

    free(bsc->data);
    bsc->data = new_data;

}

struct BSC_Int8_4in64 {
    int shape[2];
    int blocksize[2];
    int nnz_group;//nums of nonzero group
    int ncolptr;
    int8_t* data;
    GroupInfo* group_info;
    int* group_colptr;
};

BSC_Int8_4in64* create_bsc_int8_4in64(const int8_t* dense_matrix, const int shape[2]){
    int blocksize[2] = {1, 16};
    //create block=1x16 bsc
    BSCMatrix<int8_t>* bsc = create_bsc_matrix(dense_matrix, shape, blocksize);
    BSC_Int8_4in64* new_bsc = new BSC_Int8_4in64;
    new_bsc->shape[0] = shape[0];
    new_bsc->shape[1] = shape[1];
    new_bsc->blocksize[0] = blocksize[0];
    new_bsc->blocksize[1] = blocksize[1];

    new_bsc->ncolptr = bsc->ncolptr;
    new_bsc->group_colptr = new int[bsc->ncolptr];

    std::vector<GroupInfo> group_info;
    for (int b_col = 0; b_col < bsc->ncolptr - 1; ++b_col){
        new_bsc->group_colptr[b_col] = group_info.size();
        int b_row_idx = bsc->colptr[b_col];
        while (b_row_idx < bsc->colptr[b_col + 1]){
            int start_row = bsc->rowidxs[b_row_idx++];
            __mmask64 mask = 0x0000000000000001;
            int b_cnt = 1;
            while (b_cnt < 4 && b_row_idx < bsc->colptr[b_col + 1]){
                int distance = bsc->rowidxs[b_row_idx] - start_row;
                if (distance > 63){
                    break;
                }
                mask |= 1ULL << distance;
                ++b_cnt;
                ++b_row_idx;
            }
            group_info.push_back({start_row, mask});
        }
    }
    //init nnz_group,final group_colptr,group_info
    new_bsc->nnz_group = group_info.size();
    new_bsc->group_colptr[bsc->ncolptr - 1] = group_info.size();
    new_bsc->group_info = new GroupInfo[group_info.size()];
    for (int i = 0; i < new_bsc->nnz_group; ++i){
        new_bsc->group_info[i] = group_info[i];
    }

    new_bsc->data = (int8_t*) aligned_alloc(64, new_bsc->nnz_group * 4 * 16 * sizeof(int8_t));
    int data_ptr = 0;
    for (int b_col = 0; b_col < bsc->ncolptr - 1; ++b_col){
        int nnz_idx = bsc->colptr[b_col];
        for (int group_idx = new_bsc->group_colptr[b_col]; group_idx < new_bsc->group_colptr[b_col + 1]; ++group_idx){
            int valid_row_num = count1s(new_bsc->group_info[group_idx].mask);
            for (int col = 0; col < 16; ++col){
                for (int b_cnt = 0; b_cnt < valid_row_num; ++b_cnt){
                    new_bsc->data[data_ptr++] = bsc->data[(nnz_idx + b_cnt) * 16 + col];
                }
                for (int b_cnt = valid_row_num; b_cnt < 4; ++b_cnt){
                    new_bsc->data[data_ptr++] = 0;
                }
            }
            nnz_idx += valid_row_num;
        }
    }

    destroy_bsc_matrix(bsc);
    return new_bsc;
}

void destroy_bsc_int8_4in64(BSC_Int8_4in64 * bsc_int8_4in64){
    free(bsc_int8_4in64->data);
    delete[] bsc_int8_4in64->group_info;
    delete[] bsc_int8_4in64->group_colptr;
    delete bsc_int8_4in64;
}

void spdm_int8_4in64_no_cvt(
    int M, int N, int K,
    const uint8_t* A,
    const BSC_Int8_4in64* B,
    int* C
){
#define M_NBLK 4
    assert(B->blocksize[0] == 4);
    assert(B->blocksize[1] == 16);
    assert(K == B->shape[0]);
    assert(N == B->shape[1]);
    assert(K % B->blocksize[0] == 0);
    assert(N % B->blocksize[1] == 0);
    assert(M % M_NBLK == 0);

    #pragma omp parallel for collapse(2)
    for (int mb = 0; mb < M / M_NBLK; ++mb){
        for (int b_col = 0; b_col < B->ncolptr - 1; ++b_col){
            __m512i c[M_NBLK];//4x16
            for (int i = 0; i < M_NBLK; ++i){
                c[i] = _mm512_setzero_epi32();
            }
            for (int group_idx=B->group_colptr[b_col]; group_idx < B->group_colptr[b_col]+1; ++group_idx)
            {
                __m512i a[M_NBLK];
                GroupInfo group_info = B->group_info[group_idx];
                for (int i = 0; i < M_NBLK; ++i){
                    //Load 512-bits of integer data from memory (A dense) 64xint8
                    a[i] = _mm512_loadu_si512(A + (mb*M_NBLK+i)*K + group_info.start_row);
                }
                for (int i = 0; i < M_NBLK; ++i){
                    a[i] = _mm512_maskz_compress_epi8(group_info.mask, a[i]);//
                }
                for (int i = 0; i < M_NBLK; ++i){
                    //broadcast，reason of _mm512_castsi512_si128 is that the type of arg of _mm512_broadcastd_epi32 is _m128i
                    // 16 pairs 4xint8
                    a[i] = _mm512_broadcastd_epi32(_mm512_castsi512_si128(a[i]));
                }
                __m512i b = _mm512_load_epi32(&B->data[group_idx << 6]);
                for (int i = 0; i < M_NBLK; ++i){
                    // intermediate result
                    c[i] = _mm512_dpbusds_epi32(c[i], a[i], b);
                }
            }
            //store 4x16 to C corresponding position
            for (int i = 0; i < M_NBLK; ++i){
                _mm512_store_epi32(C + (mb*M_NBLK + i)*N + b_col*16, c[i]);
            }
        }
    }
}



void spdm_int8_4x16_no_cvt(
    int M, int N, int K,
    const uint8_t* A,
    const BSCMatrix<int8_t>* B,
    int* C
){
    assert(B->blocksize[0] == 4);
    assert(B->blocksize[1] == 16);
    assert(K == B->shape[0]);
    assert(N == B->shape[1]);
    assert(K % B->blocksize[0] == 0);
    assert(N % B->blocksize[1] == 0);
    assert(M % M_NBLK == 0);

    for (int mb=0; mb < M/M_NBLK, ++mb) {
        for (int b_col=0; b_col < B->ncolptr ; ++b_col){
            __m512i c[M_NBLK];
            for (int i=0; i < M_NBLK; ++i) {
                c[i] = _mm512_setzero_epi32();
            }
            for (int b_row_idx=B->colptr[b_col]; b_row_idx < B->colptr[b_col+1]; ++b_row_idx) {
                int b_row = B->rowidxs[b_row_idx];
                //handle A dense
                __m512i a[M_NBLK];
                for (int i = 0; i<M_NBLK; ++i){
                    //broadcast 32 to 512bit
                    a[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (mb*M_NBLK+i)*K + b_row));
                }
                //handle B sparse, B->data
                __m512i b = _mm512_load_epi32(&B->data[b_row_idx << 6]);
                for (int i = 0; i<M_NBLK; ++i) {
                    c[i] = _mm512_dpbusd_epi32(c[i], a[i], b);
                }
            }
            //store
            for (int i = 0; i < M_NBLK; ++i){
                _mm512_store_epi32(C + (mb*M_NBLK + i)*N + b_col*16, c[i]);
            }
        }
    }
}    

void init_input_int8_4x16(){
    const int BLK_ROW = 4;
    const int BLK_COL = 16;
    int M = 1024;
    int K = 1024;
    int N = 1024;
    srand(0);
    for (int m = 0; m < M; ++m){
        for (int k = 0; k < K; ++k){
            A[m][k] = rand() % 5 - 2;
            A_int8[m][k] = A[m][k] + 2;
        }
    }
    for (int k = 0; k < K; ++k){
        for (int n = 0; n < N; ++n){
            B[k][n] = rand() % 11 - 5;
            B_int8[k][n] = B[k][n];
        }
    }
    for (int kb = 0; kb < K / BLK_ROW; ++kb) {
        for (int nb = 0; nb < N / BLK_COL; ++nb){
            bool zero_fill = rand() % 16 != 0;
            if (zero_fill) {
                for (int k=0; k<BLK_ROW; ++k) {
                    for(int n=0; n<BLK_COL; ++n) {
                        B[kb*BLK_ROW+k][nb*BLK_COL+n] = 0;
                        B_int8[kb*BLK_ROW+k][nb*BLK_COL+n] = 0;
                    }
                }
            }
        }
    }
}

void spdm_4x16_int8(
    int M, int N, int K,
    const uint_8* A,
    const BSCMatrix<int8_t>* B,
    int* C
){
    #pragma omp parallel collapse(2)
    for (int mb=0; mb < M/M_NBLK; ++mb) {
        for (int b_col=0; b_col < B->ncolptr; ++b_col){
            __m512i c[M_NBLK];
            for (int i=0; i<M_NBLK; ++i){
                c[i] = _mm512_setzero_epi32();
            }
            for (int b_row_id=B->colptr[b_col]; b_row_id<B->colptr[b_col+1]; ++b_row_id){
                __m512i a[M_NBLK];
                int b_row = B->rowidxs[b_row_id];
                for (int i=0; i<M_NBLK; ++i) {
                    a[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (i+mb*M_NBLK)*K+b_row));
                }
                //__m512i b = B->data + blksize*b_row
                __m512i b = _mm512_load_epi32(&(B->data[b_row_id<<6]));
                for (int i=0; i<M_NBLK; ++i) {
                    c[i] = _mm512_dpbusds_epi32(c[i],a[i],b);
                }
            }
            for (int i=0; i<M_NBLK; ++i){
                _mm512_store_epi32(C+(mb*M_NBLK+i)*N, c[i]);
            }
        }
    }
}

struct BSC_Int8_4in64 {
    int shape[2];
    int blocksize[2];
    int nnz_group;
    int ncolptr;
    int8_t* data;
    GroupInfo* group_info;
    int* group_colptr;
};

void spdm_4in64_int8(
    int M, int N, int K,
    const u_int8* A,
    const BSC_Int8_4in64<int8_t>* B,
    int* C
){
    for (int mb = 0; mb<M_NBLK; ++mb) {
        for (int b_col=0; b_col<B->ncolptr; ++b_col) {
            __m512i c[M_NBLK];
            for (int i=0; i<M_NBLK; ++i) {
                c[i] = _mm512_setzero_epi32();
            }
            for (int group_idx=B->group_colptr[b_col]; group_idx<B->group_colptr[b_col+1];++group_idx) {
                GroupInfo group_info = B->group_info[group_idx];
                __m512i a[M_NBLK];
                for (int i=0; i<M_NBLK; ++i) {
                    a[i] = _mm512_loadu_si512(A + (mb*M_NBLK+i)*K + group_info.start_row);
                }
                for (int i=0; i<M_NBLK; ++i) {
                    a[i] = _mm512_maskz_compress_epi8(group_info.mask, a[i]);
                }
                for (int i=0; i<M_NBLK; ++i) {
                    a[i] = _mm512_broadcastd_epi32(_mm512_castsi512_si128(a[i]));
                }
                __m512i b = _mm512_load_epi32(&B->data[group_idx<<6]);
                for (int i = 0; i< M_NBLK; ++i) {
                    c[i] = _mm512_dpbusds_epi32(c[i], a[i], b);
                }
            }
            for (int i = 0; i < M_NBLK; ++i){
                _mm512_store_epi32(C + (mb*M_NBLK + i)*N + b_col*16, c[i]);
            }
        }
    }
}