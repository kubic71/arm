let canvas = document.createElement("canvas");
let ctx = canvas.getContext("2d");

var width = window.innerWidth * 2;
var height = window.innerHeight * 2;

canvas.width = width;
canvas.height = height;
canvas.style.width = width / 2;
canvas.style.height = height / 2;
document.body.appendChild(canvas);

ctx.fillStyle = "#ff0000";
ctx.lineWidth = 10;
ctx.font = "40px Arial";
ctx.lineJoin = "round";


function rand_normal() {
    var V1, V2, S;
    do {
        var U1 = Math.random();
        var U2 = Math.random();
        V1 = 2 * U1 - 1;
        V2 = 2 * U2 - 1;
        S = V1 * V1 + V2 * V2;
    } while (S > 1);

    X = Math.sqrt(-2 * Math.log(S) / S) * V1;

    return X;
}

function randn(rows, cols) {
    let z = math.zeros([rows, cols]);
    z = math.map(z, function () {
        return rand_normal();
    });
    return z;
}

function randn_sigma(rows, cols, sigma) {
    let z = math.zeros([rows, cols]);
    z = math.map(z, function () {
        return rand_normal() * sigma;
    });
    return z;
}

function randn_sigma_bias(rows, sigma) {
    let z = math.zeros(rows);
    z = math.map(z, function () {
        return rand_normal() * sigma;
    });
    return z;
}

function init_weights() {
    W1 = randn_sigma(DIM, DIM, .1);
    b1 = randn_sigma_bias(DIM, .1);
    W2 = randn_sigma(DIM, DIM, .1);
    b2 = randn_sigma_bias(DIM, .1);


    lengths = math.zeros(segments);
    for(let i = 0; i < segments ; i++) { 
        lengths._data[i] = math.random(400);
    }

    net = { W1: W1, b1: b1, W2: W2, b2: b2, lengths:lengths};
}

function features(target) {
    let s = { x: width / 2, y: height / 2 };
    let vect = { x: s.x - target.x, y: s.y - target.y };
    //let angle = Math.atan2(vect.y, vect.x);
    return [1, vect.x / 500, vect.y / 500];
}

var net;

function feed_forward(f, net, l, draw = true) {
    let z = math.zeros(DIM);

    for (let i = 0; i < DIM; i++) {
        if (i < f.length) {
            z._data[i] = f[i];
        } else {
            z._data[i] = 0;
        }
    }

    z = math.multiply(z, net.W1);
    z = math.add(z, net.b1)
    z = math.map(z, Math.tanh);

    z = math.multiply(z, net.W2);
    z = math.add(z, net.b2)
    z = math.map(z, Math.tanh);

    /*
    let s = {x: width/2 + z._data[0]*100, y: height/2 + z._data[1]*100};
    */

    // draw arm

    let r = math.zeros(l);
    for (let i = 0; i < l; i++) {
        r._data[i] = z._data[i] * 2 * Math.PI;
    }

    let angles = r._data;
    let s = { x: width / 2, y: height / 2 };

    if (draw) {
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        for (let i = 0; i < angles.length; i++) {
            s = { x: s.x + Math.cos(angles[i]) * (net.lengths._data[i] + 400 * z._data[segments + i]), y: s.y + Math.sin(angles[i]) * (net.lengths._data[i] + 400 * z._data[segments + i])};
            ctx.lineTo(s.x, s.y);
        }

        ctx.stroke();
        ctx.fillRect(s.x - 20, s.y - 20, 40, 40);

    } else {
        for (let i = 0; i < angles.length; i++) {
            s = { x: s.x + Math.cos(angles[i]) * (net.lengths._data[i] + 400 *z._data[segments + i]), y: s.y + Math.sin(angles[i]) * (net.lengths._data[i] + 400 * z._data[segments + i])};
        }
    }
    return s;
}

function distance_2(a, b) {
    let xd = a.x - b.x;
    let yd = a.y - b.y;
    // return Math.sqrt(xd * xd + yd * yd);
    return xd * xd + yd * yd;
}

function fitness(net, segments, n_evals=0) {
    let fit = 0

    n_evals = n_evals === 0 ? N_EVALS : n_evals;
    for (let i = 0; i < n_evals; i++) {
        // generate random target point
        target = getRandPoint();

        let f = features(target);

        let p;
        if (i === n_evals - 1) {
            // draw last eval 
            ctx.fillStyle = "#000000";
            // draw target point
            ctx.globalAlpha = 0.1;
            ctx.fillRect(target.x - 10, target.y - 10, 20, 20);
            ctx.globalAlpha = 0.02;
            // arm from weights
            ctx.fillStyle = "#0000ff";
            p = feed_forward(f, net, segments, true);
        } else {
            p = feed_forward(f, net, segments, false);
        }

        fit += -distance_2(p, target);
    }

    return fit / n_evals;
}


window.onmousemove = function (evt) {
    mouse.x = evt.offsetX * 2;
    mouse.y = evt.offsetY * 2;
};

window.onkeydown = function (evt) {
    let key = evt.key;
    if (key == "r") {
        // randomize weidhts
        init_weights();
    } else if (key == "t") {
        training = !training;
    }
}

function toggleTraining() {
    let selector = 'toggleTraining';

    training = !training;
    document.getElementById(selector).innerHTML = training ? 'Stop' : 'Train';
}

var mouse = { x: 0, y: 0 };

var segments = 6;
let DIM = 2*segments + 2; 
var net;

init_weights();

let fit = 0;


// arm

var training = false;

let learning_rate;
let exploration_sigma = 0.1; 
let sigma_auto;
let n_samples;
let N_EVALS;
let better;


function addNoise(net, noise) {
    return { W1: math.add(net.W1, noise.W1), b1: math.add(net.b1, noise.b1), W2: math.add(net.W2, noise.W2), b2: math.add(net.b2, noise.b2), lengths: math.add(net.lengths, noise.lengths)};
}

function netMultiply(net, m) {
    return { W1: math.multiply(net.W1, m), b1: math.multiply(net.b1, m), W2: math.multiply(net.W2, m), b2: math.multiply(net.b2, m), lengths:math.multiply(net.lengths, m)};
}


function zeroNet() {
    W1 = math.zeros(DIM, DIM)
    b1 = math.zeros(DIM);
    W2 = math.zeros(DIM, DIM)
    b2 = math.zeros(DIM);
    lengths = math.zeros(segments);
    return { W1: W1, b1: b1, W2: W2, b2: b2 , lengths:lengths};
}


function getRandPoint() {
    while (true) {
        let x = Math.random() * 2000- 1000;
        let y = Math.random() * 2000- 1000;

        if (x * x + y * y < 1000 * 1000) {
            return { x: width / 2 + x, y: height / 2 + y };
        }
    }
}


function update_info() {

    document.getElementById("lr_info").innerHTML = learning_rate;
    document.getElementById("sigma_info").innerHTML = exploration_sigma;
    document.getElementById("n_samples_info").innerHTML = n_samples;
    document.getElementById("n_evals_info").innerHTML = N_EVALS;
    document.getElementById("avg_fit").innerHTML = avg_fit;
}


let avg_fit = 0;

function render() {
    window.requestAnimationFrame(render);
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 10;
    ctx.fillStyle = "#000000";


    ctx.beginPath();
    ctx.arc(width / 2, height / 2, 800, 0, 2 * Math.PI);
    ctx.stroke();

    learning_rate = Math.pow(10, document.getElementById("alpha").value / 30) * 0.0001;

    sigma_auto = document.getElementById("sigma_auto").checked;
    if (!sigma_auto) {
        exploration_sigma = Math.pow(10, document.getElementById("sigma").value / 30) * 0.0001;
    }

    n_samples = document.getElementById("n_samples").value
    N_EVALS = document.getElementById("n_evals").value


    update_info();

    var rs = new Array(n_samples);

    var slider = document.getElementById("myRange");

    if (training) {
        ctx.save();
        ctx.globalAlpha = .02;

        let candidates = [];
        better = 0;
        fit = fitness(net, segments, 600);
        avg_fit = avg_fit + 0.05 * (fit - avg_fit);
        for (let i = 0; i < n_samples; i++) {


            // arm from modified weights
            let noise = {
                W1: randn_sigma(DIM, DIM, exploration_sigma), b1: randn_sigma_bias(DIM, exploration_sigma),
                W2: randn_sigma(DIM, DIM, exploration_sigma), b2: randn_sigma_bias(DIM, exploration_sigma),
                lengths: randn_sigma_bias(segments, exploration_sigma * 200),
            };

            let net_mod = addNoise(net, noise);
            ctx.fillStyle = "#ff0000";
            rs[i] = fitness(net_mod, segments)

            if (rs[i] > fit) {
                better += 1;
            }

            candidates.push(noise);
        }


        let fit_mean = math.sum(rs) / n_samples;
        let std = math.std(rs);


        let weighted_avg = zeroNet();

        for (let i = 0; i < n_samples; i++) {
            // normalized relative advantage of W_noise
            let advantage = (rs[i] - fit_mean) / std;

            // take weighted average of noise samples weighted by its advantages
            let u = netMultiply(candidates[i], advantage);
            weighted_avg = addNoise(weighted_avg, u)
        }

        var old_net = net;
        net = addNoise(net, netMultiply(weighted_avg, learning_rate / (n_samples * exploration_sigma)));

        let new_fit = fitness(old_net, segments, 600);
        if(new_fit < fit) {
            net = old_net;
        }

        // update sigma
        if (sigma_auto) {
            if (n_samples / 3 < better) {
                // if we have more than 20% better individuals
                exploration_sigma = exploration_sigma / 0.999;

            } else {
                exploration_sigma = exploration_sigma * 0.999;
            }


            exploration_sigma = Math.min(0.3, exploration_sigma);
        }

        ctx.restore();
    } else {
        ctx.globalAlpha = 1;

        let f = features(mouse);
        ctx.fillStyle = "#0000ff";
        let p = feed_forward(f, net, segments);
    }
}

render();