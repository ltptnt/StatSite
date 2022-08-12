function Picker1() {
            HideAll('1')
            Hide('1')
        }

function Picker2() {
            HideAll('2')
            Hide('2')
        }

function HideAll(id) {
            document.getElementById('picker'+id+'-Mean').style.display = 'none';
            document.getElementById('picker'+id+'-Rate').style.display = 'none';
            document.getElementById('picker'+id+'-Min').style.display = 'none';
            document.getElementById('picker'+id+'-Max').style.display = 'none';
            document.getElementById('picker'+id+'-Probability').style.display = 'none';
            document.getElementById('picker'+id+'-Trials').style.display = 'none';
            document.getElementById('picker'+id+'-Sd').style.display = 'none';
            if (document.getElementById('picker'+id+'-Output') != null) {
                document.getElementById('picker'+id+'-Output').style.display = 'none';
                document.getElementById('picker'+id+'-G_Min').style.display = 'none';
                document.getElementById('picker'+id+'-G_Max').style.display = 'none';
            }


            let labels = document.getElementsByTagName('label');
            for(let i = 0; i < labels.length; i++ ) {
                if (labels[i].htmlFor.includes('picker'+id) && !(labels[i].htmlFor.includes('Type'))) {
                    labels[i].style.display = 'none'
                }
            }
        }

function Hide(id) {
            HideAll(id);
            let name = document.getElementById('picker' + id + '-Type').options[document.getElementById('picker' + id + '-Type').selectedIndex].innerHTML
            if (name !== '') {
                const list = JSON.parse(document.getElementById(name).innerHTML.replace(/'/g, '"'));
                let labels = document.getElementsByTagName('label');

                for (let i = 0; i < list.length; i++) {
                    for (let j = 0; j < labels.length; j++) {
                        if (labels[j].htmlFor.includes('picker' + id + '-' + list[i]) ||
                            labels[j].htmlFor.includes('picker' + id + '-' + 'Output') ||
                            labels[j].htmlFor.includes('picker' + id + '-' + 'G_Min') ||
                            labels[j].htmlFor.includes('picker' + id + '-' + 'G_Max'))  {
                            labels[j].style.display = ''
                        }
                    }
                    document.getElementById('picker' + id + '-' + list[i]).style.display = '';
                }
                if (document.getElementById('picker'+id+'-Output') != null) {
                    document.getElementById('picker' + id + '-Output').style.display = '';
                    document.getElementById('picker' + id + '-G_Min').style.display = '';
                    document.getElementById('picker' + id + '-G_Max').style.display = '';
                }
            }
        }

function changeTab(tab_name) {
    let i, x, tab_buttons;
    x = document.getElementsByClassName("tab");
    for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";
    }
    tab_buttons = document.getElementsByClassName("tab_button");
    for (i = 0; i < x.length; i++) {
        tab_buttons[i].className = tab_buttons[i].className.replace(" active", "");
    }
    document.getElementById(tab_name).style.display = "block";
    document.getElementById(tab_name + "_tab").classList.add("active");
}

function openGraph(graph) {
    let newWindowContent = document.getElementById(graph).innerHTML;
    let newWindow = window.open("", "", "width=1000,height=1000");
    newWindow.document.write(newWindowContent);
}

if (document.getElementById("g1_tab") != null) {
    document.getElementById("g1_tab").click();
}

function toggle_nav() {
    const x = document.getElementById("navbar");
    if (x.style.display === "block") {
    x.style.display = "none";
  } else {
    x.style.display = "block";
  }
}


window.onload = function() {
    if (document.getElementById('picker1-Type') != null) {
        Picker1()
        Picker2()
        document.getElementById('picker1-Type').onchange = Picker1;
        document.getElementById('picker2-Type').onchange = Picker2;

        let close = document.getElementsByClassName("alert_close");
        for (let i = 0; i < close.length; i++) {
            close[i].onclick = function(){
                let div = this.parentElement;
                div.style.opacity = "0";
                setTimeout(function(){ div.style.display = "none"; }, 600);
            }
            setTimeout(function(){close[i].click()}, 10000)
         }
    }
}