// var holder = document.getElementById('drag-file');
// var path = []
// holder.ondragover = () => {
//     return false;
// };

// holder.ondragleave = () => {
//     return false;
// };

// holder.ondragend = () => {
//     return false;
// };

// holder.ondrop = (e) => {
//     e.preventDefault();

//     for (let f of e.dataTransfer.files) {
//         console.log('File(s) you dragged here: ', f.path)
//         var list_f = document.createElement('option');
//         list_f.textContent = f.path;
//         document.getElementById("l_file").appendChild(list_f)
//         path.push(f.path)         
//     }
//     console.log(path)
//     return false;
// };


function go() {

    var pyshell =  require('python-shell');
    var options = {
    args : [path] }

    pyshell.run('hello.py',options,  function  (err, results)  {
     if  (err)  throw err;
    console.log('hello.py finished.');
    console.log('results', results); });   	

}

var $ = require("jquery");
    //Canvas
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
//Variables
var canvasx = $(canvas).offset().left;
var canvasy = $(canvas).offset().top;
var last_mousex = last_mousey = 0;
var mousex = mousey = 0;
var mousedown = false;

//Mousedown
$(canvas).on('mousedown', function(e) {
    last_mousex = parseInt(e.clientX-canvasx);
	last_mousey = parseInt(e.clientY-canvasy);
    mousedown = true;
});

//Mouseup
$(canvas).on('mouseup', function(e) {
    end_x = parseInt(e.clientX-canvasx);
	end_y = parseInt(e.clientY-canvasy);
    mousedown = false;
});

//Mousemove
$(canvas).on('mousemove', function(e) {
    mousex = parseInt(e.clientX-canvasx);
	mousey = parseInt(e.clientY-canvasy);
    if(mousedown) {
        ctx.clearRect(0,0,canvas.width,canvas.height); //clear canvas
        ctx.beginPath();
        var width = mousex-last_mousex;
        var height = mousey-last_mousey;
        ctx.rect(last_mousex,last_mousey,width,height);
        ctx.strokeStyle = '#00FF2A';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    //Output
    $('#output').html('x: '+last_mousex+', '+end_x+'<br/>y: '+last_mousey+', '+end_y+'<br/>mousedown: '+mousedown);
});


function win_o(){


    
}