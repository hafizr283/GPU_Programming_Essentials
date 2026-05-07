#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

const int NUM_USERS = 5;
const int NUM_ITEMS = 5;
const int F = 2;
const double LAMBDA = 0.1;
const int ITERATIONS = 15;

// ---------------------------------------------------------
// FIX 2: Cholesky Decomposition to solve smat * x = svec
// Document বলেছে: matrix inverse না করে Cholesky ব্যবহার করো
//
// ধাপ ১: A = L * L^T (L হলো lower triangular matrix)
// ধাপ ২: L * y = b (forward substitution)
// ধাপ ৩: L^T * x = y (backward substitution)
// ---------------------------------------------------------
void cholesky_solve(double A[F][F], double b[F], double x[F]) {
    
    double L[F][F] = {};

    // ধাপ ১: Cholesky Decomposition → A = L * L^T বের করো
    for (int j = 0; j < F; j++) {
        double sum = 0;
        for (int k = 0; k < j; k++)
            sum += L[j][k] * L[j][k];
        L[j][j] = sqrt(A[j][j] - sum);

        for (int i = j + 1; i < F; i++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L[i][k] * L[j][k];
            L[i][j] = (A[i][j] - s) / L[j][j];
        }
    }

    // ধাপ ২: Forward Substitution → L * y = b সমাধান করো
    double y[F] = {};
    for (int i = 0; i < F; i++) {
        double s = 0;
        for (int k = 0; k < i; k++)
            s += L[i][k] * y[k];
        y[i] = (b[i] - s) / L[i][i];
    }

    // ধাপ ৩: Backward Substitution → L^T * x = y সমাধান করো
    for (int i = F - 1; i >= 0; i--) {
        double s = 0;
        for (int k = i + 1; k < F; k++)
            s += L[k][i] * x[k];
        x[i] = (y[i] - s) / L[i][i];
    }
}

int main() {
    double R[NUM_USERS][NUM_ITEMS] = {
        {5, 3, 0, 1, 0},
        {4, 0, 0, 1, 0},
        {1, 1, 0, 5, 4},
        {0, 0, 5, 4, 5},
        {0, 2, 4, 0, 0}
    };

    double X[NUM_USERS][F] = {
        {0.5, 0.1}, {0.2, 0.8}, {0.9, 0.3}, {0.4, 0.4}, {0.1, 0.7}
    };

    double Y[NUM_ITEMS][F] = {
        {0.1, 0.5}, {0.8, 0.2}, {0.3, 0.9}, {0.4, 0.4}, {0.7, 0.1}
    };

    cout << "Starting ALS Optimization..." << endl << endl;

    for (int iter = 1; iter <= ITERATIONS; iter++) {

        // ---------------------------------------------------------
        // STEP A: Y fix করো, X আপডেট করো
        // ---------------------------------------------------------
        for (int u = 0; u < NUM_USERS; u++) {

            double smat[F][F] = {};
            double svec[F] = {};

            for (int i = 0; i < NUM_ITEMS; i++) {
                if (R[u][i] > 0) {

                    // S1: Y^T * Y — শুধু upper triangle হিসাব করো (symmetry)
                    // FIX 1: smat[j][k] আর smat[k][j] একই, তাই loop এ অর্ধেক করি
                    for (int j = 0; j < F; j++)
                        for (int k = j; k < F; k++)  // k=j থেকে শুরু → upper triangle
                            smat[j][k] += Y[i][j] * Y[i][k];

                    // S2: Y^T * r_u
                    for (int j = 0; j < F; j++)
                        svec[j] += Y[i][j] * R[u][i];
                }
            }

            // Upper triangle থেকে lower triangle copy করো (symmetry)
            for (int j = 0; j < F; j++)
                for (int k = j + 1; k < F; k++)
                    smat[k][j] = smat[j][k];

            // Lambda * I যোগ করো (regularization)
            for (int j = 0; j < F; j++)
                smat[j][j] += LAMBDA;

            // S3: Cholesky দিয়ে smat * x_u = svec সমাধান করো
            cholesky_solve(smat, svec, X[u]);
        }

        // ---------------------------------------------------------
        // STEP B: X fix করো, Y আপডেট করো
        // ---------------------------------------------------------
        for (int i = 0; i < NUM_ITEMS; i++) {

            double smat[F][F] = {};
            double svec[F] = {};

            for (int u = 0; u < NUM_USERS; u++) {
                if (R[u][i] > 0) {

                    // S1: X^T * X — শুধু upper triangle
                    for (int j = 0; j < F; j++)
                        for (int k = j; k < F; k++)
                            smat[j][k] += X[u][j] * X[u][k];

                    // S2: X^T * r_i
                    for (int j = 0; j < F; j++)
                        svec[j] += X[u][j] * R[u][i];
                }
            }

            // Symmetry copy
            for (int j = 0; j < F; j++)
                for (int k = j + 1; k < F; k++)
                    smat[k][j] = smat[j][k];

            // Regularization
            for (int j = 0; j < F; j++)
                smat[j][j] += LAMBDA;

            // S3: Cholesky solve
            cholesky_solve(smat, svec, Y[i]);
        }
    }

    // Final Predicted Matrix R_hat = X * Y^T
    cout << "Final Predicted Matrix (R_hat):" << endl;
    cout << "[ ] = predicted (unknown),  no bracket = known rating" << endl << endl;

    for (int u = 0; u < NUM_USERS; u++) {
        for (int i = 0; i < NUM_ITEMS; i++) {
            double r_hat = 0;
            for (int f = 0; f < F; f++)
                r_hat += X[u][f] * Y[i][f];

            if (R[u][i] > 0)
                cout << fixed << setprecision(2) << r_hat << "   ";
            else
                cout << "[" << fixed << setprecision(2) << r_hat << "] ";
        }
        cout << endl;
    }

    return 0;
}