const file_re_name = []
const position = [] 
const out_po = []
const reso = []

function se(){
    const fs = require("fs");
    const {dialog} = require("electron").remote;
      dialog.showOpenDialog({
        title:"Select a folder",
        properties: ["openDirectory"]
    }, (folderPaths) => {    
      // console.log(folderPaths) 
      var pyshell =  require('python-shell');
      var options = {
      args : folderPaths }
      pyshell.run('/python_code/make_di.py',options,  function  (err, results)  {
        if  (err)  throw err;
        out_po.push(results)
      console.log(out_po)   
     });  
    });
    }


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
        pyshell.run('/python_code/re_name.py',options,  function  (err, results)  {
          if  (err)  throw err;
         file_re_name.push(results[0]);
         reso[0] = results[1];
         reso[1] = results[2];
         position[0] = 0
         position[1] = reso[1]
         position[2] = 0
         position[3] = reso[0]
         show_img()

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
       console.log("reso ",reso)
       console.log("pos ",position)
       

      

      //  console.log(file_re_name.length)
    }
    
    
    );
    
    }
    

function po(){
    // console.log(file_re_name[0])

    var pyshell =  require('python-shell');
        var options = {
        args : file_re_name[0] }
        pyshell.run('/python_code/cr.py',options,  function  (err, results)  {
          if  (err)  throw err;
          position[0] = results[0]
          position[1] = results[1]
          position[2] = results[2]
          position[3] = results[3]
          console.log(position)
       });  
       setTimeout(use_cr, 8000)
}

function start(){
  // console.log(file_re_name[0])

  var pyshell =  require('python-shell');
      var options = {
      args : [file_re_name,out_po,position[0],position[1],position[2],position[3]] }
      pyshell.run('/HelpMate.py',options,  function  (err, results)  {
        if  (err)  throw err;
        console.log(results)
        // console.log(results[1])
     });  

}


function show_img(){
  var para = document.createElement("img");
  para.setAttribute("src","./frame.png")
  para.setAttribute("id","pic")
  var element = document.getElementById("ii");
  // element.appendChild(para)
  var l = element.getElementsByTagName("img").length;
  console.log(l)

  if (l == 0)
    element.appendChild(para)
  else
    console.log("to many")
}

function use_cr(){
  var element = document.getElementById("pic");
  element.setAttribute("src","./cr.png");
  
  
}

