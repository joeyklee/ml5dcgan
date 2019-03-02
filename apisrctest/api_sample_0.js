let dcgan;
let button, outputContainer, statusMsg;

function setup() {
    noCanvas();

    //Load the model
    dcgan = ml5.DCGAN('people', modelReady);

    //status message
    statusMsg = select('#status');

    //button to generate an image
    button = select('#generate');
    button.mousePressed(generate);

    //container for the output image
    outputContainer = select('#output');
}

function generate() {
    
    dcgan.generate(canvas_element, (err, result) => {
        if (err) {
            console.log(err);
        }
        if (result && result.src) {
            // Clear output container
            outputContainer.html('');
            // Create an image based result
            createImg(result.src).parent('output');
        }
    });
}

function modelReady() {
    select('#status').html('model ready');
}
