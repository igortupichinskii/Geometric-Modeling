override group_size: u32;
override num_spline_pt: u32;
override num_ctr_pt: u32;

@binding(0) @group(0) var<storage, read> a : array<vec4f>;
@binding(1) @group(0) var<storage, read_write> res : array<vec4f>;


const knot = array<f32, 10>(
    0.0, 0.0, 0.0,
    1.0,
    2.0, 2.0,
    3.0,
    4.0, 4.0, 4.0
);

fn N0(i: u32, t: f32) -> f32 {
    if (knot[i] <= t && t < knot[i+1u]) {
        return 1.0;
    }
    return 0.0;
}

fn N1(i: u32, t: f32) -> f32 {
    var left = 0.0;
    var right = 0.0;

    let denom_left = knot[i + 1u] - knot[i];
    let denom_right = knot[i + 2u] - knot[i + 1u];

    if (denom_left != 0.0) {
        left = (t - knot[i]) / denom_left * N0(i, t);
    }

    if (denom_right != 0.0) {
        right = (knot[i + 2u] - t) / denom_right * N0(i + 1u, t);
    }

    return left + right;
}

fn N2(i: u32, t: f32) -> f32 {
    var left = 0.0;
    var right = 0.0;

    let denom_left = knot[i + 2u] - knot[i];
    let denom_right = knot[i + 3u] - knot[i + 1u];

    if (denom_left != 0.0) {
        left = (t - knot[i]) / denom_left * N1(i, t);
    }

    if (denom_right != 0.0) {
        right = (knot[i + 3u] - t) / denom_right * N1(i + 1u, t);
    }

    return left + right;
}

@compute @workgroup_size(group_size)
fn computeMain(@builtin(global_invocation_id) id : vec3<u32>) {
    
    var i : u32 = 2u;
    var j : u32;
    var t : f32;
    var n : f32;
    var num_x : f32 = 0.0;
    var den_x : f32 = 0.0;
    var num_y : f32 = 0.0;
    var den_y : f32 = 0.0;

    j = id.x;

    if (j < num_spline_pt) {
        if (j == num_spline_pt - 1u) {
            res[j] = vec4f(a[num_ctr_pt - 1u].x, a[num_ctr_pt - 1u].y, 0.0, 0.0);
            return;
        }

        t = 4.0 * f32(j) / (f32(num_spline_pt) - 1.0);

        for (var k : u32 = 2u; k <= num_ctr_pt - 1u; k = k + 1u) {
            if (t >= knot[k] && t < knot[k + 1u]) {
                i = k;
                break;
            }
        }

        for (var k : u32 = i - 2u; k <= i; k = k + 1u) {
            n = N2(k, t);
            num_x = num_x + a[k].z * a[k].x * n;
            num_y = num_y + a[k].z * a[k].y * n;
            den_x = den_x + a[k].z * n;
            den_y = den_y + a[k].z * n;
        }

        res[j] = vec4f(num_x / den_x, num_y / den_y, 0.0, 0.0);
    } 
}