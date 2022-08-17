
jack_lsp -c pure_data_0 | while read -r line; do
    #echo "prcessing $line"
    if [[ $line == pure_data_0* ]]; then
        port=$line
        #echo "$port"
    else
        jack_disconnect $port $line
    fi
done
