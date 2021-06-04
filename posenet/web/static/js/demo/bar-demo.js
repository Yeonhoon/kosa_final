Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

// var data = {{xx|safe}};
var ctx = document.getElementById('myAreaChart');
var myChart = new Chart(ctx, {
    type: 'bar',
    maintainAspectRatio: false,
    data: {
    labels: Object.values(data['DATES']),
    datasets: [{
        label: 'SQUAT VOLUME',
        data: Object.values(data['VOLUME']),
        backgroundColor: [
        'rgba(255, 99, 132, 0.2)',
        'rgba(54, 162, 235, 0.2)',
        'rgba(255, 206, 86, 0.2)',
        'rgba(75, 192, 192, 0.2)',
        'rgba(153, 102, 255, 0.2)',
        'rgba(255, 159, 64, 0.2)'
        ],
        borderColor: [
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)'
        ],
        borderWidth: 1
    }]
    },
    options: {

    scales: {
        // xAxes: [{
        //   scaleLabel : {
        //     display: true,
        //     labelString: 'Dates',
        //     fontColor : 'black'
        //   }
        // }],
        // yAxes:[{
        //   scaleLabel: {
        //     display: true,
        //     labelString: "Volume",
        //     fontColor: 'black'
        //   }
        // }],
        y: {
        beginAtZero: true
        }
    }
    }
});