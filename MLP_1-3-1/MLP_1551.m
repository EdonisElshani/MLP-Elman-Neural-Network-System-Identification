function y_hat = KNN_verbessert(set, set_val)
    
    Y = set(2:end, 2);
    X = set(1:end-1, 2);

    inputSize = 1; % Anzahl der Eingabeneuronen
    hiddenSize1 = 5; % Anzahl der Neuronen in der ersten versteckten Schicht
    hiddenSize2 = 5; % Anzahl der Neuronen in der zweiten versteckten Schicht
    outputSize = 1; % Anzahl der Ausgabeneuronen
    
    % Zufällige Gewichte und Biases initialisieren
    W1 = rand(inputSize, hiddenSize1); % Gewichte von Eingabe zu erster versteckter Schicht
    b1 = rand(1, hiddenSize1); % Biases für erste versteckte Schicht
    W2 = rand(hiddenSize1, hiddenSize2); % Gewichte von erster zu zweiter versteckter Schicht
    b2 = rand(1, hiddenSize2); % Biases für zweite versteckte Schicht
    W3 = rand(hiddenSize2, outputSize); % Gewichte von zweiter versteckter zu Ausgabeschicht
    b3 = rand(1, outputSize); % Biases für Ausgabeschicht
    
    % Lernrate
    learningRate = 0.08;
    
    i = 1;
    % Array zur Speicherung der Verlustwerte
    loss_values = zeros(1, 100);

    % Anzahl der Epochen
    epochs = 1000;
    
    % Training des MLP
    for epoch = 1:epochs
        % Vorwärtsdurchlauf
        % Erste versteckte Schicht
        Z1 = X * W1 + b1; % Lineare Kombination
        A1 = tanh(Z1); % Aktivierungsfunktion (Tanh)
        
        % Zweite versteckte Schicht
        Z2 = A1 * W2 + b2; % Lineare Kombination
        A2 = tanh(Z2); % Aktivierungsfunktion (Tanh)
    
        % Ausgabeschicht
        Z3 = A2 * W3 + b3; % Lineare Kombination (keine Aktivierungsfunktion)
    
        % Verlustfunktion (Mean Squared Error)
        loss = mean(sum((Z3 - Y).^2, 2));
    
        % Rückpropagation
        % Fehler im Ausgabelayer
        dZ3 = Z3 - Y;
        dW3 = (A2' * dZ3) / size(X, 1);
        db3 = sum(dZ3, 1) / size(X, 1);
    
        % Fehler in der zweiten versteckten Schicht
        dA2 = dZ3 * W3';
        dZ2 = dA2 .* tanh_derivative(Z2);
        dW2 = (A1' * dZ2) / size(X, 1);
        db2 = sum(dZ2, 1) / size(X, 1);
    
        % Fehler in der ersten versteckten Schicht
        dA1 = dZ2 * W2';
        dZ1 = dA1 .* tanh_derivative(Z1);
        dW1 = (X' * dZ1) / size(X, 1);
        db1 = sum(dZ1, 1) / size(X, 1);
    
        % Parameteraktualisierung
        W1 = W1 - learningRate * dW1;
        b1 = b1 - learningRate * db1;
        W2 = W2 - learningRate * dW2;
        b2 = b2 - learningRate * db2;
        W3 = W3 - learningRate * dW3;
        b3 = b3 - learningRate * db3;

       % Verlustwert alle 10 Epochen speichern
        if mod(epoch, 10) == 0
            loss_values(i) = loss;
            i = i+1;
        end
    
        % Ausgabe des Verlusts alle 100 Epochen
        if mod(epoch, 100) == 0
        %    fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end

   % Ausgabe des trainierten MLP
    figure;
    hold on; % Hold on to add multiple plots to the same figure
    
    % Plot the system data
    plot(Y, 'r', 'LineWidth', 1.5);
    
    % Plot the model data
    plot(Z3, 'b', 'LineWidth', 1.5);
    
    % Add legend
    legend('system', 'model');
    
    % Add labels and title
    xlabel('X-axis');
    ylabel('Y-axis');
    title('System vs. Model');
    
    disp('Loss Validierung:');
    disp(loss);

    % Verlustfunktion plotten
    figure;
    plot(loss_values, '-o');
    xlabel('Epoch');
    ylabel('Loss');
    title('Loss Function over Epochs');
    
    
%% Validierung

    Y_val = set_val(2:size(set_val),2);
    X_val = set_val(1:size(set_val)-1,2);
    
% Vorwärtsdurchlauf
    % Erste versteckte Schicht
    Z1 = X_val * W1 + b1; % Lineare Kombination
    A1 = tanh(Z1); % Aktivierungsfunktion (Tanh)
    
    % Zweite versteckte Schicht
    Z2 = A1 * W2 + b2; % Lineare Kombination
    A2 = tanh(Z2); % Aktivierungsfunktion (Tanh)

    % Ausgabeschicht
    Z3 = A2 * W3 + b3; % Lineare Kombination (keine Aktivierungsfunktion)


    % Verlustfunktion (Mean Squared Error)
    loss_val = mean(sum((Z3 - Y_val).^2, 2));


    % Ausgabe des trainierten MLP
    figure;
    hold on; % Hold on to add multiple plots to the same figure
    
    % Plot the system data
    plot(Y_val, 'r', 'LineWidth', 1.5);
    
    % Plot the model data
    plot(Z3, 'b', 'LineWidth', 1.5);
    
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


y_hat = Z3;

end