override group_size: u32;
// СЧИТАТЬ ПЕРЕДАННЫЕ КОНСТАНТЫ В ШЕЙДЕРЕ
override num_points_spline: u32;
override num_ctr_points: u32;

@binding(0) @group(0) var<storage, read> a : array<vec4f>;
@binding(1) @group(0) var<storage, read_write> res : array<vec4f>;

@compute @workgroup_size(group_size)
fn computeMain(@builtin(global_invocation_id) id : vec3<u32>) {

    var i : u32;
    var j : u32;
    var t : f32;
    var omega : f32;
    var x : f32;
    var y : f32;
    // РАСЧЕТ КООРДИНАТ ТОЧКИ СПЛАЙНА
    j = id.x;
    if (j >= num_points_spline) {
        return;
    }
    t = f32(j) / f32((num_points_spline - 1u));
    for (i = 0u; i < num_ctr_points - 1u; i = i + 1u){
        if (t <= a[i+1u][2u]){
            omega = (t - a[i][2u]) / (a[i+1u][2u] - a[i][2u]);
            x = a[i][0u] * (1.0 - omega) + a[i + 1u][0u] * omega;
            y = a[i][1u] * (1.0 - omega) + a[i + 1u][1u] * omega;
            res[j] = vec4f(x, y, 0.0, 0.0);
            break;
        }
    }
    // pointsSpline[id.x].x = pointsCtr[i].x * (1 - omega) + pointsCtr[i + 1].x * omega;
}