#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>
#include <vector>

// #define DEBUG

namespace
{
constexpr size_t MatSize = 15360;
constexpr int Repeat = 8;

}; // namespace

using ValueType = int;
using Mat = std::array<std::array<ValueType, MatSize>, MatSize>;
struct Task
{
    int coreNum;
    int tag;
    int x;
    int y;
    Task(int coreNum = 0, int tag = 0, int x = 0, int y = 0) : coreNum(coreNum), tag(tag), x(x), y(y)
    {
    }
};

void main_thread();
void co_thread(const int MpiRank);
void trans_mat_main(Mat &mat);
void trans_mat_co(const int MpiRank);
void init_mat(Mat &mat);
bool is_symmetrical(const Mat &A, const Mat &B);
void trans_serial(Mat &mat);

int main(int argc, char const *argv[])
{
    MPI::Init();
    auto MpiRank = MPI::COMM_WORLD.Get_rank();
    if (MpiRank == 0)
    {
        main_thread();
    }
    else
    {
        co_thread(MpiRank);
    }
    MPI::Finalize();
    return 0;
}

void main_thread()
{
    std::unique_ptr<Mat> p_mat{new Mat}, p_originMat{new Mat};
    Mat &mat = *p_mat, &originMat = *p_originMat;
#ifdef DEBUG
    std::cout << "Start init mat" << std::endl;
#endif
    init_mat(originMat);
#ifdef DEBUG
    std::cout << "Finis init mat" << std::endl;
#endif
    const int HC = MPI::COMM_WORLD.Get_size();

#ifdef DEBUG
    std::cout << "total HC:" << HC << std::endl;
#endif

    long sum = 0;
    for (int i = 0; i < Repeat; i++)
    {
        mat = originMat;
        auto t0 = std::chrono::steady_clock::now();
        if (HC > 1)
        {
            trans_mat_main(mat);
        }
        else
        {
            trans_serial(mat);
        }
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        sum += dur;
        std::cout << "HC:" << HC << "   time:" << dur << std::endl;
#ifdef DEBUG
        std::cout << (is_symmetrical(mat, originMat) ? "correct result" : "wrong result") << std::endl;
#endif
    }
    std::cout << "mean_time:" << double(sum) / Repeat << std::endl;
}

void co_thread(const int MpiRank)
{
    for (int i = 0; i < Repeat; i++)
    {
        trans_mat_co(MpiRank);
    }
}

void trans_mat_main(Mat &mat)
{
    const int HC = MPI::COMM_WORLD.Get_size();
    const int Sz = MatSize / HC;
    std::unique_ptr<int[]> srbuf{new int[Sz * Sz]};
    std::vector<Task> taskVec;
    int msgTag = 0;

    auto recvFunc = [&]() {
        for (auto it = taskVec.begin(); it != taskVec.end(); ++it)
        {
#ifdef DEBUG
            std::cout << "Start Recv Line:" << __LINE__ << "    "
                      << "from:" << it->coreNum << "    "
                      << "thread:" << MPI::COMM_WORLD.Get_rank() << "    Tag:" << it->tag << std::endl;
#endif
            MPI::COMM_WORLD.Recv(srbuf.get(), Sz * Sz, MPI::INT, it->coreNum, it->tag);
#ifdef DEBUG
            std::cout << "Finis Recv Line:" << __LINE__ << "    "
                      << "from:" << it->coreNum << "    "
                      << "thread:" << MPI::COMM_WORLD.Get_rank() << std::endl;
#endif
            int(*arrPtr)[Sz] = reinterpret_cast<int(*)[Sz]>(srbuf.get());
            for (int r = 0; r < Sz; r++)
            {
                std::copy_n(arrPtr[r], Sz, mat[it->x + r].begin() + it->y);
            }
        }
        taskVec.clear();
    };

    for (int bi = 0, coreNum = 0; bi < MatSize; bi += Sz)
    {
        for (int bj = 0; bj < MatSize; bj += Sz)
        {
            if (++coreNum == HC)
            {
                recvFunc();
                coreNum = 1;
            }

            for (int r = 0; r < Sz; r++)
            {
                std::copy_n(mat[bi + r].begin() + bj, Sz, srbuf.get() + r * Sz);
            }
#ifdef DEBUG
            std::cout << "Start Send Line:" << __LINE__ << "    "
                      << "to:" << coreNum << "    "
                      << "thread:" << MPI::COMM_WORLD.Get_rank() << "    Tag:" << msgTag << std::endl;
#endif
            MPI::COMM_WORLD.Send(srbuf.get(), Sz * Sz, MPI::INT, coreNum, msgTag);
#ifdef DEBUG
            std::cout << "Finis Send Line:" << __LINE__ << "    "
                      << "to:" << coreNum << "    "
                      << "thread:" << MPI::COMM_WORLD.Get_rank() << std::endl;
#endif
            taskVec.emplace_back(coreNum, msgTag++, bi, bj);
        }
    }
    recvFunc();
    for (int bi = Sz, coreNum = 0; bi < MatSize; bi += Sz)
    {
        for (int bj = 0; bj < bi; bj += Sz)
        {
            for (int r = 0; r < Sz; r++)
            {
                std::swap_ranges(mat[bi + r].begin() + bj, mat[bi + r].begin() + bj + Sz, mat[bj + r].begin() + bi);
            }
        }
    }
#ifdef DEBUG
    std::cout << "exit thread:" << MPI::COMM_WORLD.Get_rank() << std::endl;
#endif
}

void trans_mat_co(const int MpiRank)
{
    const int HC = MPI::COMM_WORLD.Get_size();
    const int Sz = MatSize / HC;
    std::unique_ptr<int[]> srbuf{new int[Sz * Sz]};
    int taskCnt = HC + 1 + (MpiRank == 1);
    for (int cnt = 0; cnt < taskCnt; cnt++)
    {
#ifdef DEBUG
        std::cout << "Start Recv Line:" << __LINE__ << "    "
                  << "thread:" << MPI::COMM_WORLD.Get_rank() << "    Tag:" << cnt * (HC - 1) + MpiRank - 1 << std::endl;
#endif
        MPI::COMM_WORLD.Recv(srbuf.get(), Sz * Sz, MPI::INT, 0, cnt * (HC - 1) + MpiRank - 1);
#ifdef DEBUG
        std::cout << "Finis Recv Line:" << __LINE__ << "    "
                  << "thread:" << MPI::COMM_WORLD.Get_rank() << std::endl;
#endif

        int(*arrPtr)[Sz] = reinterpret_cast<int(*)[Sz]>(srbuf.get());
        for (int i = 1; i < Sz; i++)
        {
            for (int j = 0; j < i; j++)
            {
                auto tmp = arrPtr[i][j];
                arrPtr[i][j] = arrPtr[j][i];
                arrPtr[j][i] = tmp;
            }
        }
#ifdef DEBUG
        std::cout << "Start Send Line:" << __LINE__ << "    "
                  << "thread:" << MPI::COMM_WORLD.Get_rank() << "    Tag:" << cnt * (HC - 1) + MpiRank - 1 << std::endl;
#endif
        MPI::COMM_WORLD.Send(srbuf.get(), Sz * Sz, MPI::INT, 0, cnt * (HC - 1) + MpiRank - 1);
#ifdef DEBUG
        std::cout << "Finis Send Line:" << __LINE__ << "    "
                  << "thread:" << MPI::COMM_WORLD.Get_rank() << std::endl;
#endif
    }
#ifdef DEBUG
    std::cout << "exit thread:" << MPI::COMM_WORLD.Get_rank() << std::endl;
#endif
}

void init_mat(Mat &mat)
{
    std::default_random_engine gen{std::random_device{}()};
    for (auto &v : mat)
    {
        for (auto &i : v)
        {
            i = gen();
        }
    }
}

bool is_symmetrical(const Mat &A, const Mat &B)
{
    for (size_t i = 0; i < A.size(); i++)
    {
        for (size_t j = 0; j < B.size(); j++)
        {
            if (A[i][j] != B[j][i])
            {
                std::cout << "A[" << i << "][" << j << "]=" << A[i][j] << "  B[" << j << "][" << i << "]=" << B[j][i]
                          << std::endl;
                return false;
            }
        }
    }
    return true;
}

void trans_serial(Mat &mat)
{
    for (int i = 1; i < mat.size(); i++)
    {
        for (int j = 0; j < i; j++)
        {
            auto tmp = mat[i][j];
            mat[i][j] = mat[j][i];
            mat[j][i] = tmp;
        }
    }
}