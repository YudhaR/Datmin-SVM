// Function to initialize DataTable
const initializeDataTable = async (dataSourceURL, tableID, tableBodyID) => {
    let dataTable;
    let dataTableIsInitialized = false;

    const dataTableOptions = {
        pageLength: 4,
        destroy: false
    };
    
    if (tableID !== 'datatable-trains' && tableID !== 'datatable-tests') {
        dataTableOptions.columnDefs = [
            { className: "centered", targets: [0, 1, 2, 3] },
            { orderable: false, targets: [3] },
            { searchable: false, targets: [3] }
        ];
    } else if (tableID === 'datatable-trains') {
        dataTableOptions.columnDefs = [
            { className: "centered", targets: [0, 1, 2, 3, 4, 5, 6] },
            { orderable: false, targets: [6] },
            { searchable: false, targets: [6] }
        ];
    } else if (tableID === 'datatable-tests') {
        dataTableOptions.columnDefs = [
            { className: "centered", targets: [0, 1, 2] },
            { orderable: false, targets: [2] },
            { searchable: false, targets: [2] }
        ];
    }

    const listData = async () => {
        try {
            // Fetch data
            const response = await fetch(dataSourceURL);
            const data = await response.json();

            // Create HTML content for table rows
            let content = "";
            data.items.forEach((item, index) => {
                content += `
                    <tr>
                        <td>${index + 1}</td>
                        ${
                            tableID != 'datatable-tests' ?
                            `
                            <td>${item.fields.username || ''}</td>
                            ` :
                            ''
                        }
                        <td>${item.fields.full_text || ''}</td>
                        ${
                            tableID == 'datatable-trains' ?
                            `
                                <td>${item.fields.compound_score}</td>
                                <td>${item.fields.sentiment}</td> 
                                <td>${item.fields.depresi}</td>
                            ` :
                            ''
                        }
                        ${
                            tableID == 'datatable-tests' ?
                            `
                                <td>${item.fields.depresi}</td>
                            ` :
                            ''
                        }
                        ${
                            tableID != 'datatable-tests' ?
                            `
                            <td>
                                <a href='/delete/list-tweet/${item.fields.idt}' class='btn btn-sm btn-danger'><i class='fa-solid fa-trash-can'></i></a>
                             </td>
                            ` :
                            ''
                        }
                    </tr>`;
            });

            // Update the table body
            $(`#${tableBodyID}`).html(content);
        } catch (ex) {
            alert(ex);
        }
    };

    const initDataTable = async () => {
        try {
            // Destroy existing instance if initialized
            if (dataTableIsInitialized) {
                dataTable.destroy();
            }

            // Fetch and display data
            await listData();

            // Initialize DataTable
            dataTable = $(`#${tableID}`).DataTable(dataTableOptions);

            // Mark as initialized
            dataTableIsInitialized = true;
        } catch (ex) {
            alert(ex);
        }
    };

    // Ensure the window event listener is asynchronous
    window.addEventListener("load", async () => {
        await initDataTable();
    });
};

// Example usage for tweets
initializeDataTable('http://127.0.0.1:8000/list-tweet/', 'datatable-tweets', 'tableBody_tweet');

// Example usage for cleans
initializeDataTable('http://127.0.0.1:8000/list-clean/', 'datatable-cleans', 'tableBody_clean');

initializeDataTable('http://127.0.0.1:8000/list-training/', 'datatable-trains', 'tableBody_train');

initializeDataTable('http://127.0.0.1:8000/list-testing/', 'datatable-tests', 'tableBody_test');
