function myFunction() {
    var ss = SpreadsheetApp.openByUrl("https://docs.google.com/spreadsheets/d/1RlHEqXJqVD0NhWL-EqFHdqE9G_TYS0qFVfkuwSq0ErQ/edit");
    var sheet = ss.getSheetByName("t");
    var Data = sheet.getRange(1,1).getValue();
    // Google Calendar ーID
    var calId = "a73nb9q00738qr7t2kpe7icej4@group.calendar.google.com";
    // LINE Notify Token
    var key = "iDT4ySiVipep5Nnr3TgD3uupYh2cn6iypc98sQNg1Qz";
                                  
    var url = "https://notify-api.line.me/api/notify";
  
    var cal = CalendarApp.getCalendarById(calId);
    var now = new Date();
    var today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    // Google Calendar รับกำหนดการในวันพรุ่งนี้จากปฏิทินเป็นอาร์เรย์
    var todayEvent = cal.getEventsForDay(today);
  
    // LINE Notify เพื่อส่งไปยัง
    var msg = "";
    // เมื่อไม่มีกิจกรรมที่เกิดขึ้น
      msg = Utilities.formatDate(today,"JST","yyyy/MM/dd")+"\n"+"วันนี้้เวรกลุ่มที่"+(Data%3+1)+"\n"+g[Data%3]+"\n"+"ทำเวรด้วย!!!!!!!!!!!!!!!";
      sheet.getRange(1,1).setValue(Data+1);
  
    var jsonData = {
      message: msg
    }
  
    var options =
    {
      "method" : "post",
      "contentType" : "application/x-www-form-urlencoded",
      "payload" : jsonData,
      "headers": {"Authorization": "Bearer " + key}
    };
  
    var res = UrlFetchApp.fetch(url, options);
  }
  
  // ส่งคืนอาร์เรย์ของเหตุการณ์เป็นข้อความ
  function allPlanToMsg(events/* array */){
    var msg = "";
    events.forEach( function(event, index){
      var title = event.getTitle();
      var start = event.getStartTime().getHours() + ":" + ("0" + event.getStartTime().getMinutes()).slice(-2);
      var end = event.getEndTime().getHours() + ":" + ("0" + event.getEndTime().getMinutes()).slice(-2);
      // เมื่อถึงกำหนดวันทั้งวัน
      if( event.isAllDayEvent() ){
        msg += String(index + 1) + ": " + title + g+"  \n\n";
        return;
      }
      msg += String(index + 1) + ": " + title + " " + start + "~" + end + "\n";
    });
    return msg;
  } 

  var g = ['Bas4D,Two4D,Nine4D,Time4D,Saimai3D,Pakbung3D,Pose3D,Get2D,Ming2D,In1D,Tom1D,Pom1D,Few1D',
            'Tin4D,Farnce4D,Frong4D,Tle3D,Nonny3D,Jik3D,Tintin3D,Aomsin3D,Muay3D,Oat1D,Tan1D,Boss1D',
            'Poom4D,Mixz4D,Aomsin4D,JJ3D,Earth3D,Nine3D,Beb3D,Oeng3D,Kee2D,Tew2D,Fang2D,Net2D,Pond1D,Mail1D,Poom1D'];