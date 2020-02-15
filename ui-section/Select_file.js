const file_re_name = []

function fi(){
const fs = require("fs");
const {dialog} = require("electron").remote;
  dialog.showOpenDialog({
    title:"Select a folder",
     filters: [
    { name: 'Movies', extensions: ['mkv', 'avi', 'mp4'] },
    
  ],
  properties: ['openFile','multiSelections']
}, (folderPaths) => {
   for(var i=0;i<folderPaths.length;i++){
    var pyshell =  require('python-shell');
    var options = {
    args : folderPaths[i] }
    pyshell.run('re_name.py',options,  function  (err, results)  {
      if  (err)  throw err;
     file_re_name.push(results);
    //  file_list.push(results[i])
     
   });  

    // file_list.push(folderPaths[i])
    var iframe = document.getElementById("bg");
    // var elmnt = iframe.contentWindow.document.getElementById("f");
    var h = iframe.contentWindow.document.getElementById("f");
    var tr = document.createElement("tr")
    var th_path = document.createElement("th")
    th_path.innerText = folderPaths[i]
    h.appendChild(tr)
    tr.appendChild(th_path)
   }
  //  console.log("this is path : ",file_list)
   console.log("rename ",file_re_name)
  //  console.log(file_re_name.length)
}


);
}
