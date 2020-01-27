const {app, BrowserWindow} = require('electron')

function createWindow () {
    window = new BrowserWindow({width: 1000, height: 800})
    window.loadFile('index.html')


    	/*var python = require('child_process').spawn('python', ['./hello.py']);
	python.stdout.on('data',function(data){
    		console.log("data: ",data.toString('utf8'));
	});*/


// var pyshell =  require('python-shell');

// var options = {
//   // scriptPath : path.join(__dirname, '/../engine/'),
//   args : ['asdasdasdasd']
// }


// pyshell.run('hello.py',options,  function  (err, results)  {
//  if  (err)  throw err;
//  console.log('hello.py finished.');
//  console.log('results', results);
// });   	
  

}




app.on('ready', createWindow)

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit()
    }
})

