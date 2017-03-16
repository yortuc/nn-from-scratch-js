/************************ MAT FUNCTIONS *********************************/
function prod_vec(v1, v2){
	var ret = 0;
	for(var i=0; i<v1.length; i++){
		ret += v1[i] * v2[i];
	}
	return ret;
}

function test_prod_vec(){
	var v1 = [1,2,3];
	var v2 = [5,6,7];
	var result = 5+12+21;

	console.log(prod_vec(v1,v2), result);
}

function had_product(m1, m2){
	// check dimensions
	// m1_x = m2_y && m1_y = m2_x
	if(m1[0].length !== m2.length){
		throw new Error("Cannot had_product. matrix dimensions dont match.");
	}

	var ret = [];

	for(var i=0; i<m1.length; i++){
		var ret_row = [];

		var m1_row = m1[i];

		for(var j=0; j<m2[0].length; j++){
			
			var m2_col = [];
			for(var k=0; k<m2.length; k++){
				m2_col.push(m2[k][j])
			}

			ret_row.push(prod_vec(m1_row, m2_col));
		}
		ret.push(ret_row);
	}

	return ret;
}

function test_had_product(){
	const m1 = [[1,2,3],
			    [2,3,4]];

	const m2 = [[4,5],
			    [6,7],
			    [8,9]];

	const result = [[40, 46],
					[58, 67]];

	console.log(had_product(m1,m2), result);
}

function add_scalar_to_vec(v, s){
	var ret = [];
	for(var i=0; i<v.length; i++){
		ret.push(v[i] + s);
	}
	return ret;
}

function test_add_scalar_to_vec(){
	const v = [1,
			   2,
			   3,
			   4,
			   5];
	const s = 2;
	const result = [3,4,5,6,7];
	console.log(add_scalar_to_vec(v,s), result);
}

/*
 *	componentwise product btw two equal-sized matrices 
 *	(Hadamard product)
 */
function comp_prod(m1,m2){
	// multiply component-wise
	const ret = m1.map((row,i)=> row.map((col,j)=> m1[i][j]*m2[i][j]));
	return ret;
}

function test_comp_prod(){
	const m1 = [[1,2],
				[2,3]];

	const m2 = [[4,5],
				[6,7]];

	const result = [[ 4,10],
					[12,21]];
	console.log(comp_prod(m1,m2));
	console.log(result);
}

function mat_transpose(m){
	const ret = m[0].map((col, i)=> m.map(row=> row[i]));
	return ret;
}

function test_mat_transpose(){
	const m = [[1,2,3],
			   [4,5,6]];

	const result = [[1,4],
					[2,5],
					[3,6]];
	console.log(mat_transpose(m));
	console.log(result);
}

function mat_add(m1,m2){
	const ret = m1.map((row,i)=> row.map((col,j)=> m1[i][j] + m2[i][j]));
	return ret;
}

function test_mat_add(){
	const m1 = [[1,2,3], [3,4,5]];
	const m2 = [[-1,-2,-3], [-3,-4,-5]];
	const result = [[0,0,0],[0,0,0]];
	console.log(mat_add(m1,m2));
	console.log(result);
}

function mat_mul_with_scalar(m, s){
	return m.map(row=> row.map(col=> col*s));
}

/************************ READ INPUT *********************************/

function read_input_file(callback, filename="./data/iris_training.dat") {
	// Read the file and print its contents.
	var fs = require('fs');
	
	fs.readFile(filename, 'utf8', function(err, data) {
	  if (err) throw err;
	  callback(data);
	});
}

// return {x:[[]], y:[[]]}
function parse_input_data(data){
	var x = [];
	var y = [];

	data.split('\n').forEach(row=> {
		if(row.length>0){
			var r_x = [];
			var r_y = [];
 			row.split('\t').forEach((col, index)=> {
				if(index<4) r_x.push(parseFloat(col));
				if(index>=4) r_y.push(parseFloat(col));
			});
			x.push(r_x);
			y.push(r_y);
		}
	});

	return {x, y};
}

/************************ UTILITY FUNCTIONS *********************************/

function log_mat(caption, m){
	return;
	var ret = "";

	for(var i=0; i<m.length; i++){
		var row = "";
		for(var j=0; j<m[0].length; j++){
			row += m[i][j].toFixed(5) + " ";
		}
		ret += "   " + row + "\n";
	}
	console.log("----------------------");
	console.log(caption + " ("+m[0].length + "x" + m.length+"):");
	console.log(ret);
	console.log("----------------------");
}

function shuffle_arrays(x, y){
	var currentIndex = x.length, tmp1, tmp2, randomIndex;

	// While there remain elements to shuffle...
	while (0 !== currentIndex) {
		// Pick a remaining element...
		randomIndex = Math.floor(Math.random() * currentIndex);
		currentIndex -= 1;

		// And swap it with the current element.
		tmp1 = x[currentIndex];
		tmp2 = y[currentIndex];
		x[currentIndex] = x[randomIndex];
		y[currentIndex] = y[randomIndex];
		x[randomIndex] = tmp1;
		y[randomIndex] = tmp2;
	}

	return {x, y};
}

function test_shuffle_arrays(){
	var x = [1,2,3,4,5];
	var y = [2,4,6,8,10];
	console.log({x,y});
	var shuffled = shuffle_arrays(x,y);
	console.log(shuffled);
}

/************************ NEURAL NETWORK *********************************/

function init_weights(li_count, lh_count, lo_count){
	// init W1 matrix
	// x:input layer neuron count, y: hidden layer neuron count

	const max = 0.05;
	const min = -0.05;
	const computeParam = ()=> Math.random(); //(max-min)*Math.random() + min;

	const w1 = [];
	for(var i=0; i<li_count; i++){
		let row = [];
		for(var j=0; j<lh_count; j++){
			row.push(computeParam());
		}
		w1.push(row);
	}

	// init w2 matrix
	// x: hidden layer neuron count, y: output layer neuron count
	const w2 = [];
	for(var i=0; i<lh_count; i++){
		let row = [];
		for(var j=0; j<lo_count; j++){
			row.push(computeParam());
		}
		w2.push(row);
	}

	const b1 = [];
	for(var j=0; j<lh_count; j++){
		b1.push(computeParam());
	}

	const b2 = [];
	for(var j=0; j<lo_count; j++){
		b2.push(computeParam());
	}

	return {w1, b1, w2, b2};
}

function layer_z(input, weights, bias) {
	// compute z
	const z_prod = had_product(input, weights);
	const z = z_prod.map(row=> row.map((col,i)=> col + bias[i]));

	return z;
}

function hyperbolic_tangent(m){
	const ht = (x)=> Math.tanh(x);

	// matrice iterate
	var ret = [];
	for(var i=0; i<m.length; i++){
		var row = [];
		for(var j=0; j<m[0].length; j++){
			row.push(ht(m[i][j]));
		}
		ret.push(row);
	}
	return ret;
}

function hyperbolic_tangent_grad(m){
	const ht_grad = (y)=> (1-Math.pow(Math.tanh(y),2));

	// matrice iterate
	var ret = [];
	for(var i=0; i<m.length; i++){
		var row = [];
		for(var j=0; j<m[0].length; j++){
			row.push(ht_grad(m[i][j]));
		}
		ret.push(row);
	}
	return ret;
}

function softmax(m) {
	// not to get Math.exp overflow, use the trick that in softmax, 
	// only the distance btw points matters
	// softmax([1001,1002]) = softmax([-1,0])
	
	let ret = m.map(row=> {
		let max_in_row = Math.max(...row);
		let row_normalized = row.map(r=> r - max_in_row);
		let denom = row_normalized.reduce((prev, current)=> prev + Math.exp(current), 0);
		let row_smax = row_normalized.map(c=> Math.exp(c)/denom);
		return row_smax;
	});

	return ret;
}

function test_softmax(){
	const m = [[1001,1002],[3,4]];
	const result = [[0.26894142, 0.73105858], [0.26894142, 0.73105858]];

	console.log(softmax(m));
	console.log(result);
}

function softmax_grad(m) {
	// softmax' = y * (1-y)
	const ret = m.map(row=> row.map(col=> col * (1-col)));
	return ret;
}

/**************************** TRAINING *********************************/

function feed_forward(x, w1, b1, w2, b2, y) {
	const z1 = layer_z(x, w1, b1);
	const a1 = hyperbolic_tangent(z1);

	log_mat("z1", z1);
	log_mat("a1", a1);

	const z2 = layer_z(z1, w2, b2);
	const a2 = softmax(z2);	// output

	log_mat("z2", z2);
	log_mat("a2", a2);

	return {a1, a2};
}

var error_history = [];

function back_prop(w2, a1, y_pred, y){
	// grad w2
	const diff_output = y.map((row,i)=> row.map((col,j)=> y[i][j] - y_pred[i][j]));	// diff = y - y_pred
	const deriv_a2 = softmax_grad(y_pred);
	const grad_output = comp_prod(diff_output, deriv_a2);	// output gradient = (y-y_pred) * softmax'(a2)
	log_mat("d2", grad_output);

	// grad b2 (b2: [l2_neuron_count])
	const init_b2_grad = [];
	for(var i=0; i<y_pred[0].length; i++){ init_b2_grad.push(0) }
	const grad_b2 = diff_output.reduce((curRow, prevRow)=> curRow.map((cCol, i)=> curRow[i] + prevRow[i]), init_b2_grad);

	// grad w1
	const sum = had_product(grad_output, mat_transpose(w2));
	const deriv_a1 = hyperbolic_tangent_grad(a1);
	const grad_hidden = comp_prod(sum, deriv_a1);
	log_mat("d1", grad_hidden);

	// grad b1 (py: gradb1 = np.sum(delta, axis = 0)
	const init_b1_grad = [];
	for(var i=0; i<grad_hidden[0].length; i++){ init_b1_grad.push(0) }
	const grad_b1 = grad_hidden.reduce((curRow, prevRow)=> curRow.map((cCol, i)=> curRow[i] + prevRow[i]), init_b1_grad);

	// cost (mean squares)
	const error = diff_output.map(row=> row.reduce((prev, cur)=> prev + Math.pow(cur,2), 0));
	const mse = (error.reduce((p,c)=>c+p,0))/y.length;
	console.log("error: ", mse);
	error_history.push(mse);

	return {d1: grad_hidden, db1: grad_b1, d2: grad_output, db2: grad_b2, error};
}

function update_weights(x, w1, b1, a1, d1, db1, w2, b2, a2, d2, db2, learningRate=0.05, momentum=0.01){
	// update w1 = w1 + x d1 * ep
	const w1_updated = mat_add(w1, mat_mul_with_scalar(had_product(mat_transpose(x), d1), learningRate));
	log_mat("new w1: ", w1_updated);

	// update b1
	const b1_updated = b1.map((b,i)=> b + b * db1[i] * learningRate);

	// update w2
	const w2_updated = mat_add(w2, mat_mul_with_scalar(had_product(mat_transpose(a1), d2), learningRate));
	log_mat("new w2: ", w2_updated);

	// update b2
	const b2_updated = b2.map((b,i)=> b + b * db2[i] * learningRate);
	last_b2_grad = db2;

	return {w1: w1_updated, b1: b1_updated, w2: w2_updated, b2: b2_updated};
}

function iterate_training(x, w1, b1, w2, b2, y, it_count, it_max){
	const {a1, a2} = feed_forward(x, w1, b1, w2, b2, y);
	const {d1, db1, d2, db2} = back_prop(w2, a1, a2, y);

	const updated_weights = update_weights(x, w1, b1, a1, d1, db1, w2, b2, a2, d2, db2);

	if(it_count < it_max) {
		iterate_training(x, updated_weights.w1, updated_weights.b1, updated_weights.w2, updated_weights.b2, y, it_count+1, it_max);
	}
	else{
		end_training(w1, b1, w2, b2, a2, y);
	}
}

function iterate_training_incremental(x, w1, b1, w2, b2, y){
 	const {a1, a2} = feed_forward(x, w1, b1, w2, b2, y);
	const {d1, db1, d2, db2, error} = back_prop(w2, a1, a2, y);

	const updated_weights = update_weights(x, w1, b1, a1, d1, db1, w2, b2, a2, d2, db2);
	return {w1: updated_weights.w1, b1: updated_weights.b1, w2: updated_weights.w2, b2:updated_weights.b2, a2, error};
}

function end_training(w1, b1, w2, b2, a2, y) {
	// show final weights and bias values
	console.log("------------ training ended ------------");
	console.log("predictions \t\t\t data");

	var ret = "";
	for(var i=0; i<a2.length; i++){
		let row = "";
		for(var j=0; j<a2[0].length; j++){
			row += a2[i][j].toFixed(3) + "\t";
		}
		row += "\t";
		for(var j=0; j<a2[0].length; j++){
			row += y[i][j] + "\t";
		}
		ret += row + "\n";
	}
	console.log(ret);
	console.log("------- accuracy --------");
	console.log("% ", compute_accuracy(a2, y));
	console.log("------ validation -------");
	compute_validation_error(w1,b1,w2,b2);
}


function compute_accuracy(a2, y) {
	var numCorrect = 0,
		numWrong = 0;

	a2.map((row,i)=> {
		let maxIndex = row.indexOf(Math.max(...row));

		if(y[i][maxIndex] === 0) {
			numWrong += 1;
		}
		else{
			numCorrect += 1;
		}
	});

	console.log(numCorrect + " correct ", numWrong + " wrong");

	return 100*numCorrect/y.length;
}

function compute_validation_error(w1,b1,w2,b2){
	// load valdation data
	read_input_file((rawData)=> {
		const data = parse_input_data(rawData);
		const {a2} = iterate_training_incremental(data.x, w1, b1, w2, b2, data.y);

		console.log("% ", compute_accuracy(a2, data.y));

	}, "./data/iris_validation.dat");
}

// test_prod_vec();
// test_had_product();
// test_add_scalar_to_vec();
// test_comp_prod();
// test_mat_transpose();
// test_mat_add();
// test_softmax();
// test_shuffle_arrays();

function test_train_nn(){
	// test nn with an arbitrary input and output

	const x = [[1,2,3]];		// input
	const y = [[0.25, 0.75]];	// target
	var {w1, b1, w2, b2} = init_weights(3,4,2);

	iterate_training(x, w1, b1, w2, b2, y, 0, 10);
}

function iris_train_nn(){
	read_input_file(rawData => {
		const {x, y} = parse_input_data(rawData);
		const {w1, b1, w2, b2} = init_weights(4,7,3);

		iterate_training(x, w1, b1, w2, b2, y, 0, 1000);
	});
}

const fs = require('fs');

function iris_train_incremental(maxEpochs=1000, mse=0.04){
	var stop_training = false;

	read_input_file(rawData => {
		const data = parse_input_data(rawData);
		var {w1, b1, w2, b2} = init_weights(4,7,3);
		var epoch = 0, updated_weights;

		while(epoch<maxEpochs){	
			let shuffled_data = shuffle_arrays(data.x, data.y);

			for(var i=0; i<data.y.length; i++){
				// update weights for each data item
				updated_weights = iterate_training_incremental([shuffled_data.x[i]], w1, b1, w2, b2, [shuffled_data.y[i]]);
				w1 = updated_weights.w1;
				b1 = updated_weights.b1;
				w2 = updated_weights.w2;
				b2 = updated_weights.b2;

				/*if(updated_weights.error < 0.00000001) {
					console.log("\t *** error achieved");
					stop_training = true;
					break;
				}*/
			}

			if(stop_training) break;

			epoch += 1;
		}

		const {a2} = iterate_training_incremental(data.x, w1, b1, w2, b2, data.y);
		end_training(w1,b1,w2,b2,a2, data.y);
		fs.writeFile('error_history.txt', error_history.join('\n'), function (err) { console.log("log saved.") });
			
	});
}

// test_train_nn();
// iris_train_nn();
iris_train_incremental();


