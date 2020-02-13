var holder = document.getElementById('drag-file');
// let $ = require("jquery") 

  holder.ondragover = () => {
      return false;
  };

  holder.ondragleave = () => {
      return false;
  };

  holder.ondragend = () => {
      return false;
  };

  holder.ondrop = (e) => {
      e.preventDefault();

      for (let f of e.dataTransfer.files) {
          console.log('File(s) you dragged here: ', f.path)
          var h = document.getElementById("f")
          var tr = document.createElement("tr")
          var th_name = document.createElement("th")
          th_name.innerText = f.name
          var th_path = document.createElement("th")
          th_name.setAttribute("onclick","gg(this.innerText)")
          th_path.setAttribute("onclick","gg(this.innerText)")
          th_path.innerText = f.path
          h.appendChild(tr)
          tr.appendChild(th_name)
          tr.appendChild(th_path)
      }
      
      return false;
  };




function gg(x){
    console.log(x)
}



// $(document).ready(function(){
//   $("#drag-file tbody").click(function(){
//     var t_data = $(this).children("td").map(function(){
//       return $(this).text();
//     }).get();
//     var td=t_data[0]
//     console.log(td)
//   });
// });
// }