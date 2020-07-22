

async function run(){
    const csvURL = 'iris.csv';

    const trainingData = tf.data.csv(csvURL, {
        columnConfigs: {
            species: {
                isLabel: true
            }
        }
    });

    const numOfFeatures = (await trainingData.columnNames()).length - 1;
    const numOfSamples = 150;

    //converting dictionary returned to trainingData arrays
    //and applying one-hot encoding to the labels
    const convertData = trainingData.map(({xs, ys}) =>{
        const labels = [
            ys.species == "setosa" ? 1 : 0,
            ys.species == "virginica" ? 1:0,
            ys.species == "versicolor" ? 1:0]

        return{ xs: Object.values(xs), ys: Object.values(labels)};

    }).batch(10);


    const model = tf.sequential();
    //creating a model with one hidden layer with softmax activation
    model.add(tf.layers.dense({inputShape: [numOfFeatures], activation: "sigmoid", units: 5}))
    model.add(tf.layers.dense({units: 3, activation:"softmax"}));

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.06)
    })

    await model.fitDataset(
        convertData,
        {
            epochs:100,
            callbacks:{
                onEpochEnd: async(epoch, logs) =>{
                    console.log("Epoch: " + epoch + " Loss: " + logs.loss);
                }
            }
        }
    )

    const testVal = tf.tensor2d([5.8,4,1.2,0.2], [1, 4]);
    const prediction = model.predict(testVal);
    alert(prediction);
}

run();