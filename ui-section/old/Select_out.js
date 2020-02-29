
function se(){
const fs = require("fs");
const {dialog} = require("electron").remote;
  dialog.showOpenDialog({
    title:"Select a folder",
    properties: ["openDirectory"]
}, (folderPaths) => {    
  console.log(folderPaths) 
  var pyshell =  require('python-shell');
  var options = {
  args : folderPaths }
  pyshell.run('make_di.py',options,  function  (err, results)  {
    if  (err)  throw err;
  console.log(results)   
 });  

});
  
}
