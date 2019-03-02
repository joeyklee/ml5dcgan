let dcgan;
let outputCanvas, button, statusMsg;

function setup() {

    //load the model
    //we can have multiple pre-trained models (e.g. cats, flowers, etc.), just like SketchRNN
    dcgan = ml5.DCGAN('people', modelReady);

    //status message
    statusMsg = select('#status');

    //button to generate an image
    button = select('#generate');
    button.mousePressed(generate);

    //canvas for the output image
    outputCanvas = select('#canvas');
}

function generate() {
    //the generate function takes an output canvas to draw on
    //and a callback with possible info like time elapsed to generate the image
    dcgan.generate(outputCanvas, () => {
        //some callback
    });
}

function modelReady() {
    select('#status').html('model ready');
}
