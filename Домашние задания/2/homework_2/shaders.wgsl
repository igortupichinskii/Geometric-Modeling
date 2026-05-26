@binding(0) @group(0) var<storage, read_write> a_mat : array<f32>;
@binding(1) @group(0) var<storage, read_write> x_vec : array<f32>;
@binding(2) @group(0) var<storage, read_write> b_vec : array<f32>;

override matrix_dim: u32;
override eps: f32;


var<workgroup> temp: array<f32, matrix_dim>;
var<workgroup> residual: f32;

@compute @workgroup_size(matrix_dim)
fn computeMain(@builtin(global_invocation_id) gid : vec3<u32>) {
    var tid : u32 = gid.x;
    var row_id : u32 = tid * matrix_dim;
    var i : u32;
    var t : f32;
    b_vec[tid] = b_vec[tid] / a_mat[row_id + tid];
    storageBarrier();
    for (var j : u32 = 0u; j < matrix_dim; j = j + 1u){
        if (j != tid) {
            a_mat[row_id + j] = -a_mat[row_id + j] / a_mat[row_id + tid];
        }
    }
    x_vec[tid] = 1.0;
    if (tid == 0u) {
        residual = 0.0;
    }
    workgroupBarrier();
    storageBarrier();
    for (var iter : u32 = 0; iter < 1000u; iter = iter + 1u) {
        if (tid == 0u) {
            residual = 0.0;
        }
        workgroupBarrier();
        //Вычисляем слагаемые i-ой компонентны нового вектора x^(k+1), заносим в массив temp
        for (i = 0u; i < matrix_dim; i = i + 1){
            if (tid != i) {
                temp[tid] = a_mat[i * matrix_dim + tid] * x_vec[tid];
            } else {
                temp[tid] = b_vec[tid];
            }
            workgroupBarrier();
            var shift : u32 = 1u;
            var k : u32 = 2u;
            loop {
                if (shift > matrix_dim) {
                    break;
                }

                if (tid % k == 0u) {
                    if (tid + shift < matrix_dim) {
                        temp[tid] = temp[tid] + temp[tid + shift];
                    }
                }
                workgroupBarrier();

                shift = shift * 2u;
                k = k * 2u;
            }
            //посчитать приращение компоненты
            if (tid == 0) {
                t = abs(x_vec[i] - temp[0]);
                x_vec[i] = temp[0];
            }
            if (t > residual) {
                residual = t;
            }
            storageBarrier();
            
        }
        workgroupBarrier();
        let r : f32 = workgroupUniformLoad(&residual);
        if (r <= eps) {
            break;
        }
    }
}
