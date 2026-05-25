@binding(0) @group(0) var<storage, read_write> a_mat : array<f32>;
@binding(1) @group(0) var<storage, read_write> x_vec : array<f32>;
@binding(2) @group(0) var<storage, read_write> b_vec : array<f32>;

override matrix_dim: u32;


var<workgroup> max_val: array<f32, matrix_dim>;
var<workgroup> max_row: array<u32, matrix_dim>;
var<workgroup> order: array<u32, matrix_dim>;
@compute @workgroup_size(matrix_dim)
fn computeMain(@builtin(global_invocation_id) gid : vec3<u32>) {
    var tid : u32 = gid.x;
    var i : u32;
    var shift : u32;
    var k : f32;
    var max_row_id : u32;
    order[tid] = matrix_dim - 1u;
    workgroupBarrier();
    //Прямой ход
    //Цикл по столбцам
    for (i = 0u; i < matrix_dim; i = i + 1u){
        // Определяем строку с максимальным элементом на i-ом столбце
        if (order[tid] == matrix_dim - 1u) {
            max_val[tid] = abs(a_mat[tid * matrix_dim + i]);
            max_row[tid] = tid;
        } else {
            max_val[tid] = -1.0;
        }
        

        workgroupBarrier();

        shift = matrix_dim / 2u;

        loop {
            if (shift == 0u) {
                break;
            }

            if (tid < shift) {
                if (max_val[tid] < max_val[tid + shift]) {
                    max_val[tid] = max_val[tid + shift];
                    max_row[tid] = max_row[tid + shift];
                }
            }
            workgroupBarrier();

            shift = shift / 2u;
        }
        max_row_id = max_row[0];

        if (tid != max_row_id) {
            if (order[tid] == matrix_dim - 1u) {
                k = a_mat[tid * matrix_dim + i] / a_mat[max_row_id * matrix_dim + i];
            a_mat[tid * matrix_dim + i] = 0.0;
            for (var j : u32 = i + 1u; j < matrix_dim; j = j + 1u) {
                a_mat[tid * matrix_dim + j] = a_mat[tid * matrix_dim + j] - k * a_mat[max_row_id * matrix_dim + j];
            }
            b_vec[tid] = b_vec[tid] - k * b_vec[max_row_id];
            }
        } else {
            order[tid] = i;
        }

        storageBarrier();
    }
    // Обратный ход
    for (var step : u32 = 0u; step < matrix_dim; step = step + 1u) {
        var j : u32 = matrix_dim - 1u - step;
        if (order[tid] == j) {
            x_vec[j] = b_vec[tid] / a_mat[tid * matrix_dim + j];
        }
        storageBarrier();
        if (order[tid] != j) {
            b_vec[tid] = b_vec[tid] - x_vec[j] * a_mat[tid * matrix_dim + j];
        }
        storageBarrier();
    }
    workgroupBarrier();
    storageBarrier();
}


/*
@compute @workgroup_size(matrix_dim, matrix_dim)
fn computeMain(@builtin(global_invocation_id) gid : vec3<u32>) {

}
*/