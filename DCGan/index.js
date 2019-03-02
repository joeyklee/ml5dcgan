// --------------------------------------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------------------------------------
let all_model_info = {
    dcgan64: {
        description: 'DCGAN, 64x64 (16 MB)',
        model_url: "model/model.json",
        model_size: 64,
        model_latent_dim: 128,
        draw_multiplier: 4,
        animate_frame: 200,
    },
    resnet128: {
        description: 'ResNet, 128x128 (252 MB)',
        model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-resent128-celebahq-128/tfjs_SmoothedGenerator_20000/model.json",
        model_size: 128,
        model_latent_dim: 128,
        draw_multiplier: 2,
        animate_frame: 10
    },
    resnet256: {
        description: 'ResNet, 256x256 (252 MB)',
        model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_40000/model.json",
        model_size: 256,
        model_latent_dim: 128,
        draw_multiplier: 1,
        animate_frame: 10
    }
};
let default_model_name = 'dcgan64';
function image_enlarge(y, draw_multiplier) {
    if (draw_multiplier === 1) {
        return y;
    }
    let size = y.shape[0];
    return y.expandDims(2).tile([1, 1, draw_multiplier, 1]
    ).reshape([size, size * draw_multiplier, 3]
    ).expandDims(1).tile([1, draw_multiplier, 1, 1]
    ).reshape([size * draw_multiplier, size * draw_multiplier, 3])
}

function computing_prep_canvas(size) {
    // We don't `return tf.image.resizeBilinear(v1, [size * draw_multiplier, size * draw_multiplier]);`
    // since that makes image blurred, which is not what we want.
    // So instead, we manually enlarge the image.
    let canvas = document.getElementById("the_canvas");
    let ctx = canvas.getContext("2d");
    ctx.canvas.width = size;
    ctx.canvas.height = size;
}

function callCallback(promise, callback) {
    if (callback) {
        promise
            .then((result) => {
                callback(undefined, result);
                return result;
            })
            .catch((error) => {
                callback(error);
                return error;
            });
    }
    return promise;
}

class DCGAN{
    constructor(model_name, ready_cb){
        this.model_promise_cache = {};
        this.model_promise = null;
        this.model_name = model_name;
        this.model = null;
        this.start_time = null;
        this.canvas_to_draw = null;
        this.ready = callCallback(this.loadModel(), ready_cb);
    }

    async loadModel() {
        let model_name = this.model_name;
        let model_info = all_model_info[model_name];
        let model_size = model_info.model_size,
            model_url = model_info.model_url,
            draw_multiplier = model_info.draw_multiplier,
            description = model_info.description;

        computing_prep_canvas(model_size * draw_multiplier);
        // ui_set_canvas_wrapper_size(model_size * draw_multiplier);
        // ui_logging_set_text(`Setting up model ${description}...`);

        if (model_name in this.model_promise_cache) {
            this.model_promise = this.model_promise_cache[model_name];
            // ui_logging_set_text(`Model "${description}" is ready.`);
            return this;
        } else {
            // ui_generate_button_disable('Loading...');
            // ui_animate_button_disable('Loading...');
            // ui_logging_set_text(`Loading model "${description}"...`);
            this.model_promise = await tf.loadModel(model_url);
            this.model_promise_cache[model_name] = this.model_promise;
            console.log("model setup!");
            console.log(this.model_promise_cache[model_name]);
            return this;
            // this.model_promise.then((model) => {
            //     return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
            // }).then((model) => {
            //     ui_generate_button_enable();
            //     ui_animate_button_enable();
            //     ui_logging_set_text(`Model "${description}" is ready.`);
            // });
        }
    }

    async generate(inputElement, cb){
        return callCallback(this.generate_internal(inputElement), cb);
    }

    async generate_internal(inputElement) {
        let model_info = all_model_info[this.model_name];
        let model_size = model_info.model_size,
            model_latent_dim = model_info.model_latent_dim,
            draw_multiplier = model_info.draw_multiplier;

        computing_prep_canvas(model_size * draw_multiplier);
        // ui_logging_set_text('Generating image...');
        // ui_generate_button_disable('Generating...');
        // ui_animate_button_disable();

        // this.model_promise.then((model) => {
        //     return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
        // }).then((model) => {
        //     this.start_ms = (new Date()).getTime();
        //     return computing_generate_main(model, model_size, draw_multiplier, model_latent_dim);
        // }).then((_) => {
        //     let end_ms = (new Date()).getTime();
        //     ui_generate_button_enable();
        //     ui_animate_button_enable();
        //     ui_logging_set_text(`Image generated. It took ${(end_ms - this.start_ms)} ms.`);
        // });
        this.start_time = (new Date()).getTime();
        //await computing_generate_main(this., model_size, draw_multiplier, model_latent_dim);
        let model = await this.model_promise;
        await this.computing_generate_main(model, model_size, draw_multiplier, model_latent_dim, inputElement);
        let end_ms = (new Date()).getTime();
        return end_ms - this.start_time;
    }

    async computing_generate_main(model, size, draw_multiplier, latent_dim, inputElement) {
        const y = tf.tidy(() => {
            const z = tf.randomNormal([1, latent_dim]);
            const y = model.predict(z).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(0.5));
            return image_enlarge(y, draw_multiplier);

        });
        await tf.toPixels(y, inputElement);
    }

}

let canvas = document.getElementById('the_canvas');
let dcgan = new DCGAN("dcgan64",cbb);

function cbb(){
    dcgan.generate(canvas, (result) =>{console.log(result)});
}