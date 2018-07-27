let data,xs,ys;
let model;
let graphX = [];
let graphY = [];
let rSlider, gSlider, bSlider;

let labelList = [
	'red-ish',
	'green-ish',
	'blue-ish',
	'orange-ish',
	'yellow-ish',
	'pink-ish',
	'purple-ish',
	'brown-ish',
	'grey-ish'
]
function preload(){
	data = loadJSON('colorData.json');		//preloading data
}

function setup() {
	labelP = createP('label:');
	epochP = createP('Epoch');
	lossP = createP('Loss');
	var myCanvas = createCanvas(300,150);
	myCanvas.position(500, 425);
	rSlider = createSlider(0, 255, 255);
	gSlider = createSlider(0, 255, 255);
	bSlider = createSlider(0, 255, 0);

	let colors = [];
	let labels = [];

	for (let record of data.entries){
		let col = [record.r/255, record.g/255, record.b/255];
		colors.push(col);
		labels.push(labelList.indexOf(record.label));
	}

	xs = tf.tensor2d(colors);
	let labelsTensor = tf.tensor1d(labels, 'int32');

	ys = tf.oneHot(labelsTensor, 9);
	labelsTensor.dispose();

	model = tf.sequential();

	let hidden = tf.layers.dense({																													//initializing and compiling the model
		units: 16,
		activation: 'sigmoid',
		inputDim: 3
	});

	let output = tf.layers.dense({
		units: 9,
		activation: 'softmax'
	});

	model.add(hidden);
	model.add(output);

	const learningRate = 0.2;
	const optimizer = tf.train.sgd(learningRate);

	model.compile({
		optimizer: optimizer,
		loss: 'categoricalCrossentropy',
	});

	train();

}

async function train(){

	let options = {
		epochs: 10,
		validationSplit: 0.1,
		shuffle: true,
		callbacks: {
			// onTrainBegin: () => console.log('training start'),
			// onTrainEnd: () => console.log('training end'),
			onBatchEnd: () => tf.nextFrame(),
			onEpochEnd: (num,logs) => {																													//fitting the model with callbacks
				epochP.html('Epoch: '+ num),
				graphX.push(num);
				graphY.push(logs.loss);
				lossP.html('Loss: '+ logs.loss);

			}
		}
	}
	return await model.fit(xs,ys, options);
}




function draw() { //visualisation of loss and prediction
	let r = rSlider.value()
	let g = gSlider.value()
	let b = bSlider.value()
	background(r,g,b);

	var lossfunc = {
		x: graphX, //epochs
		y: graphY, // loss
		type: 'scatter'
	};
	var data = [lossfunc];
	Plotly.newPlot('myDiv', data);
				tf.tidy(() => {
					const xs = tf.tensor2d([
						[r/255,g/255,b/255]
					]);
					let results = model.predict(xs);
					let index = results.argMax(1).dataSync()[0];
					let label = labelList[index];

					labelP.html(label);

				});
}
