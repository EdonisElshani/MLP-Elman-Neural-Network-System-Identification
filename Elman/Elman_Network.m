function y_hat = Elman_Network(set,set_val)
    
    Y = set(2:end, 2);
    X = set(1:end-1, 2);

    i=1;
    inputSize = 1; % Anzahl der Eingabeneuronen
    hiddenSize = 5; % Anzahl der Neuronen in der versteckten Schicht
    outputSize = 1; % Anzahl der Ausgabeneuronen
    
    % Zufällige Gewichte und Biases initialisieren
    W1 = rand(inputSize + hiddenSize, hiddenSize); % Gewichte von Eingabe und Kontextneuronen zu versteckter Schicht
    b1 = rand(1, hiddenSize); % Biases für versteckte Schicht
    W2 = rand(hiddenSize, outputSize); % Gewichte von versteckter zu Ausgabeschicht
    b2 = rand(1, outputSize); % Biases für Ausgabeschicht
    
    % Lernrate
    learningRate = 0.1;
    
    % Array zur Speicherung der Verlustwerte
    loss_values = zeros(1, 100);

    % Anzahl der Epochen
    epochs = 1000;
    
    % Initialisierung der Kontextneuronen
    context = zeros(length(X), hiddenSize);
    
    % Training des Elman-Netzwerks
    for epoch = 1:epochs
        
        % Vorwärtsdurchlauf
        % Versteckte Schicht
        input = [X, context]; % Eingabe und Kontextneuronen kombinieren
        Z1 = input * W1 + b1; % Lineare Kombination
        A1 = tanh(Z1); % Aktivierungsfunktion (Tanh)
    
        % Ausgabeschicht
        Z2 = A1 * W2 + b2; % Lineare Kombination (keine Aktivierungsfunktion)
    
        % Verlustfunktion (Mean Squared Error)
        loss = mean(sum((Z2 - Y).^2, 2));
        
    
        % Rückpropagation
        % Fehler im Ausgabelayer
        dZ2 = Z2 - Y;
        dW2 = (A1' * dZ2) / length(X);
        db2 = sum(dZ2, 1) / length(X);
    
        % Fehler im versteckten Layer
        dA1 = dZ2 * W2';
        dZ1 = dA1 .* tanh_derivative(Z1);
        dW1 = (input' * dZ1) / length(X);
        db1 = sum(dZ1, 1) / length(X);
    
        % Parameteraktualisierung
        W1 = W1 - learningRate * dW1;
        b1 = b1 - learningRate * db1;
        W2 = W2 - learningRate * dW2;
        b2 = b2 - learningRate * db2;
    
        % Kontextneuronen aktualisieren
        context = A1;
    
        % Verlustwert alle 10 Epochen speichern
        if mod(epoch, 10) == 0
            loss_values(i) = loss;
            i = i+1;
        end
    end
    
    % Ausgabe des trainierten Elman-Netzwerks
    disp('Trainiertes Elman-Netzwerk:');
    disp(Z2);
    
    % Plotten der System- und Modelldaten
    figure;
    hold on; % Hold on to add multiple plots to the same figure
    
    % Plot the system data
    plot(Y, 'r', 'LineWidth', 1.5);
    
    % Plot the model data
    plot(Z2, 'b', 'LineWidth', 1.5);
    
    % Add legend
    legend('system', 'model');
    
    % Add labels and title
    xlabel('X-axis');
    ylabel('Y-axis');
    title('System vs. Model');
    
    % Verlustfunktion plotten
    figure;
    plot(10:10:epochs, loss_values, '-o');
    xlabel('Epoch');
    ylabel('Loss');
    title('Loss Function over Epochs');
    

    %% Valiierung


    Y_val = set_val(2:size(set_val),2);
    X_val = set_val(1:size(set_val)-1,2);

    % Vorwärtsdurchlauf
    % Versteckte Schicht
    input = [X_val, context]; % Eingabe und Kontextneuronen kombinieren
    Z1 = input * W1 + b1; % Lineare Kombination
    A1 = tanh(Z1); % Aktivierungsfunktion (Tanh)

    % Ausgabeschicht
    Z2 = A1 * W2 + b2; % Lineare Kombination (keine Aktivierungsfunktion)

    % Verlustfunktion (Mean Squared Error)
    loss_val = mean(sum((Z2 - Y_val).^2, 2));

     % Ausgabe des trainierten MLP
    figure;
    hold on; % Hold on to add multiple plots to the same figure
    
    % Plot the system data
    plot(Y_val, 'r', 'LineWidth', 1.5);
    
    % Plot the model data
    plot(Z2, 'b', 'LineWidth', 1.5);
    
    % Add legend
    legend('system', 'model');
    
    % Add labels and title
    xlabel('Time');
    ylabel('Rad');
    title('System vs. Model (Validierung)');
    
    disp('Loss Validierung:');
    disp(loss_val);


    % Tanh-Aktivierungsfunktion
    function y = tanh(x)
        y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
    end
    
    % Ableitung der Tanh-Aktivierungsfunktion
    function y = tanh_derivative(x)
        y = 1 - tanh(x).^2;
    end


y_hat = Z2;

end