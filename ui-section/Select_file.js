const file_list = []


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
  // let a = folderPaths
  // console.log(folderPaths)
  // console.log(folderPaths[2])
   for(var i=0;i<folderPaths.length;i++){
    file_list.push(folderPaths[i])
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
}


);
}
