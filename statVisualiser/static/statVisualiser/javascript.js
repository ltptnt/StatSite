function Picker1(){HideAll("1"),Hide("1")}function Picker2(){HideAll("2"),Hide("2")}function HideAll(e){document.getElementById("picker"+e+"-Mean").style.display="none",document.getElementById("picker"+e+"-Rate").style.display="none",document.getElementById("picker"+e+"-Min").style.display="none",document.getElementById("picker"+e+"-Max").style.display="none",document.getElementById("picker"+e+"-Probability").style.display="none",document.getElementById("picker"+e+"-Trials").style.display="none",document.getElementById("picker"+e+"-Sd").style.display="none",null!==document.getElementById("picker"+e+"-Output")&&(document.getElementById("picker"+e+"-Output").style.display="none",document.getElementById("picker"+e+"-G_Min").style.display="none",document.getElementById("picker"+e+"-G_Max").style.display="none");for(var t=document.getElementsByTagName("label"),n=0;n<t.length;n++)t[n].htmlFor.includes("picker"+e)&&!t[n].htmlFor.includes("Type")&&(t[n].style.display="none")}function Hide(e){HideAll(e);var t=document.getElementById("picker"+e+"-Type").options[document.getElementById("picker"+e+"-Type").selectedIndex].innerHTML;if(""!==t){const n=JSON.parse(document.getElementById(t).innerHTML.replace(/'/g,'"'));for(var l=document.getElementsByTagName("label"),c=0;c<n.length;c++){for(var d=0;d<l.length;d++)(l[d].htmlFor.includes("picker"+e+"-"+n[c])||l[d].htmlFor.includes("picker"+e+"-Output")||l[d].htmlFor.includes("picker"+e+"-G_Min")||l[d].htmlFor.includes("picker"+e+"-G_Max"))&&(l[d].style.display="");document.getElementById("picker"+e+"-"+n[c]).style.display=""}null!==document.getElementById("picker"+e+"-Output")&&(document.getElementById("picker"+e+"-Output").style.display="",document.getElementById("picker"+e+"-G_Min").style.display="",document.getElementById("picker"+e+"-G_Max").style.display="")}}function changeTab(e){var t,n,l;for(n=document.getElementsByClassName("tab"),t=0;t<n.length;t++)n[t].style.display="none";for(l=document.getElementsByClassName("tab_button"),t=0;t<n.length;t++)l[t].className=l[t].className.replace(" active","");document.getElementById(e).style.display="block",document.getElementById(e+"_tab").classList.add("active")}function openGraph(e){var t=document.getElementById(e).innerHTML,n=window.open("","","width=1000,height=1000");n.document.write(t)}function toggle_nav(){const e=document.getElementById("navbar");"block"===e.style.display?e.style.display="none":e.style.display="block"}null!==document.getElementById("g1_tab")&&document.getElementById("g1_tab").click(),window.onload=function(){if(null!==document.getElementById("picker1-Type")){Picker1(),Picker2(),document.getElementById("picker1-Type").onchange=Picker1,document.getElementById("picker2-Type").onchange=Picker2;for(var e=document.getElementsByClassName("alert_close"),t=0;t<e.length;t++)e[t].onclick=function(){var e=this.parentElement;e.style.opacity="0",setTimeout(function(){e.style.display="none"},300)},setTimeout(function(){e[t].click()},8e3)}};